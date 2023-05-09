from sklearn.preprocessing import QuantileTransformer
import logging
import sys
import yaml
import torch
import pandas as pd
import seaborn as sns
import numpy as np
import xarray as xr
import os
from fmcml.model import DNN, get_device
from fmcml.seed import seed_everything
from fmcml.plot import (
    plot_site_vars,
    calc_site_r2_rmse,
    calc_skill_scores,
    calc_skill_scores_site,
    drop_outliers,
)
from fmcml.data import load_data, load_splitter, vegetatation_indices
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt
import joblib
from itertools import product
import warnings

# print("Installing xarray...")
# !{sys.executable} -m pip -q install "xarray[complete]"
# print("xarray install complete")
# print("Installing geopandas...")
# !{sys.executable} -m pip -q install geopandas
# print("geopandas install complete")
# print("Installing sklearn...")
# !{sys.executable} -m pip -q install scikit-learn
# print("sklearn install complete")

warnings.filterwarnings("ignore")


def metrics(
    model,
    x,
    y,
    scaler_y=None,
    torch_model=False,
    device="cpu",
    batch_size=128,
    clim=None,
):
    if torch_model:
        model.eval()
        with torch.no_grad():
            X = np.array_split(x, x.shape[0] / batch_size)
            pred = torch.cat(
                [model(torch.from_numpy(_x).to(device).float()).cpu() for _x in X]
            )
    else:
        pred = model.predict(x)
        if len(pred.shape) == 1:
            pred = np.expand_dims(model.predict(x), 1)
    truth = y
    if scaler_y:
        pred = scaler_y.inverse_transform(pred)
        truth = scaler_y.inverse_transform(truth)
    metrics_dict = {}
    metrics_dict["truth"] = truth
    metrics_dict["pred"] = pred
    metrics_dict["rmse"] = mean_squared_error(truth, pred) ** (1 / 2)
    metrics_dict["mae"] = np.mean(np.abs(truth - pred))
    metrics_dict["r2_score"] = r2_score(truth, pred)
    metrics_dict["mape"] = np.mean(abs(truth - pred) / (0.5 * (truth + pred) + 1e-8))
    metrics_dict["diff_mean"] = np.mean(truth) - np.mean(pred)

    #     if not isinstance(clim, None):
    try:
        for col in clim.columns:
            t = col.split("-")[-1]
            c = np.isfinite(clim[col].values)
            logging.info(f"Fraction of df used in {t} climatology metrics: {c.mean()}")
            metrics_dict[f"rmse_{t}"] = mean_squared_error(
                truth[c][:, 0], clim[col][c].values
            ) ** (1 / 2)
            metrics_dict[f"mae_{t}"] = np.mean(
                np.abs(truth[c][:, 0] - clim[col][c].values)
            )
            metrics_dict[f"r2_score_{t}"] = r2_score(
                truth[c][:, 0], clim[col][c].values
            )
            metrics_dict[f"mape_{t}"] = np.mean(
                abs(truth[c][:, 0] - clim[col][c].values)
                / (0.5 * (truth[c][:, 0] + clim[col][c].values) + 1e-8)
            )
            metrics_dict[f"diff_mean_{t}"] = np.mean(truth[c][:, 0]) - np.mean(
                clim[col][c].values
            )
            metrics_dict[f"skill_score_rmse_{t}"] = (
                1 - metrics_dict["rmse"] / metrics_dict[f"rmse_{t}"]
            )
            metrics_dict[f"skill_score_mae_{t}"] = (
                1 - metrics_dict["mae"] / metrics_dict[f"mae_{t}"]
            )
            metrics_dict[f"skill_score_r2_score_{t}"] = (
                metrics_dict["r2_score"] - metrics_dict[f"r2_score_{t}"]
            ) / (1 - metrics_dict[f"r2_score_{t}"])
            metrics_dict[f"skill_score_mape_{t}"] = (
                1 - metrics_dict["mape"] / metrics_dict[f"mape_{t}"]
            )
            metrics_dict[f"skill_score_diff_mean_{t}"] = 1 - abs(
                metrics_dict["diff_mean"] / metrics_dict[f"diff_mean_{t}"]
            )
    except:
        pass
    return metrics_dict


def evaluate(
    conf, climatology_fn="/glade/scratch/jimenez/tmp/jpss_datasets/madis_clim.nc", 
    load = False
):

    seed = conf["seed"]
    seed_everything(seed)
    verbose = conf["verbose"]
    save_loc = conf["save_loc"]
    data_loc = conf["data_loc"]
    total_input_vars = []
    if conf["use_nwm"]:
        total_input_vars += conf["nwm_vars"]
    if conf["use_sat"]:
        total_input_vars += conf["sat_vars"]
    if conf["use_static"]:
        total_input_vars += conf["static_vars"]
    if conf["use_hrrr"]:
        total_input_vars += conf["hrrr_vars"]
    if conf["use_lst"]:
        total_input_vars += conf["lst_vars"]
    output_vars = conf["output_vars"]
    device = get_device()

    if "model" != conf and "n_splits" in conf:
        model_type = "lg"
        n_splits = conf["n_splits"]
    elif "n_splits" in conf["model"]:
        model_type = "xgb"
        n_splits = conf["model"]["n_splits"]
    else:
        model_type = "mlp"
        n_splits = conf["trainer"]["n_splits"]

    # Load the data and take the first split
    # Load the data frame with filtered columns
    if "csv" in data_loc:
        df = pd.read_csv(data_loc)
        if "use_veg" in conf:
            if conf["use_veg"]:
                logging.info("Adding vegetation indices computed using VIIRS")
                df, veg_indices = vegetatation_indices(df)
                total_input_vars += veg_indices
        filter_nans = []
        if isinstance(total_input_vars, list):
            filter_nans += total_input_vars
        if isinstance(output_vars, list):
            filter_nans += output_vars
        if len(filter_nans):
            if verbose:
                logging.info(f"Starting df size: {df.shape}")
                logging.info(f"Filtering NaNs from columns: {filter_nans}")
            filter_condition = df[filter_nans].isna().sum(axis=1).astype(bool)
            df = df[~filter_condition].copy()
            logging.info(f"Training df size after removing NaNs: {df.shape}")

        if "other_vars" in conf:
            if conf["use_other"]:
                for var in conf["other_vars"]:
                    if var == "doy":
                        logging.info("Adding DOY to the list of predictors")
                        df["doy"] = df["Times"].astype("datetime64").dt.dayofyear  # .astype(float)
                        total_input_vars.append("doy")
                    else:
                        logging.info(f"Adding {var} to the list of predictors")
                        total_input_vars.append(var)
                        #total_input_vars.append("latitude")
                        #total_input_vars.append("longitude")

    else:
        df = load_data(
            data_loc,
            verbose=verbose,
            filter_input_vars=total_input_vars,
            filter_output_vars=output_vars,
        )
        # Add hourly climatology to the dataframe
        logging.info("Adding climatology to the dead FMC dataframe")
        df["day"] = pd.to_datetime(df["n_time"]).dt.dayofyear
        df["hour"] = pd.to_datetime(df["n_time"]).dt.hour
        clim_df = xr.open_dataset(climatology_fn)
        day, hour, site = zip(
            *list(product(clim_df["JDay"].values, clim_df["HR"].values, range(1799)))
        )
        merge_df = pd.DataFrame.from_dict(
            {
                "day": day,
                "hour": hour,
                "n_site": site,
                "climatology-hour": clim_df["DOY_HR_FMC_Climo_375m"]
                .values[:, :, :, 0]
                .ravel(),
            }
        )
        climatology = df[["day", "hour", "n_site"]].merge(
            merge_df, on=["day", "hour", "n_site"], how="left"
        )
        df["climatology-hour"] = climatology["climatology-hour"]
        # Add daily climatology to the dataframe
        day, site = zip(*list(product(clim_df["JDay"].values, range(1799))))
        daily_df = pd.DataFrame.from_dict(
            {
                "day": day,
                "n_site": site,
                "climatology-day": clim_df["DOY_FMC_Climo_375m"]
                .values[:, :, 0]
                .ravel(),
            }
        )
        climatology = df[["day", "n_site"]].merge(
            daily_df, on=["day", "n_site"], how="left"
        )
        df["climatology-day"] = climatology["climatology-day"]

    data_folds = load_splitter(
        conf["split_type"], df, n_splits=n_splits, seed=seed, verbose=conf["verbose"]
    )

    """
        Evaluation
    """

    if model_type == "mlp":
        torch_model = True
        # Load the model and weights
        input_size = len(total_input_vars)
        middle_size = conf["model"]["middle_size"]
        output_size = len(output_vars)
        dropout = conf["model"]["dropout"]
        num_layers = conf["model"]["num_layers"]

        model = DNN(
            input_size,
            output_size,
            block_sizes=[middle_size for k in range(num_layers)],
            dr=[dropout for k in range(num_layers)],
        ).to(device)

        checkpoint = torch.load(
            f"{save_loc}/best.pt", map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        with open(os.path.join(conf["save_loc"], "scalers.pkl"), "rb") as fid:
            scaler_x, scaler_y, best_fold = joblib.load(fid)

    else:
        torch_model = False
        with open(os.path.join(conf["save_loc"], "model_results.pkl"), "rb") as fid:
            model, _ = joblib.load(fid)
        with open(os.path.join(conf["save_loc"], "scalers.pkl"), "rb") as fid:
            scaler_x, scaler_y, best_fold = joblib.load(fid)
            
    for k_fold, (train_data, valid_data, test_data) in enumerate(data_folds):
        
        if k_fold != best_fold:
            continue
        
#         scaler_x = QuantileTransformer(
#             n_quantiles=1000, random_state=seed, output_distribution="normal"
#         )  # StandardScaler()
#         scaler_y = QuantileTransformer(
#             n_quantiles=1000, random_state=seed, output_distribution="normal"
#         )  # StandardScaler()
        
        x_train = scaler_x.transform(train_data[total_input_vars])
        x_valid = scaler_x.transform(valid_data[total_input_vars])
        x_test = scaler_x.transform(test_data[total_input_vars])
        y_train = scaler_y.transform(train_data[output_vars])
        y_valid = scaler_y.transform(valid_data[output_vars])
        y_test = scaler_y.transform(test_data[output_vars])
        break

    climatologies = ["climatology-hour", "climatology-day"]
    clim_col = climatologies[0]

    # Compute metrics
    if clim_col in train_data:
        y_train_clim = train_data[climatologies]
        y_valid_clim = valid_data[climatologies]
        y_test_clim = test_data[climatologies]
    else:
        y_train_clim = False
        y_valid_clim = False
        y_test_clim = False

    train_metrics = metrics(
        model,
        x_train,
        y_train,
        scaler_y,
        torch_model=torch_model,
        device=device,
        clim=y_train_clim,
    )
    valid_metrics = metrics(
        model,
        x_valid,
        y_valid,
        scaler_y,
        torch_model=torch_model,
        device=device,
        clim=y_valid_clim,
    )
    test_metrics = metrics(
        model,
        x_test,
        y_test,
        scaler_y,
        torch_model=torch_model,
        device=device,
        clim=y_test_clim,
    )

    # Save metric scores
    scores = ["rmse", "r2_score", "mape", "mae", "diff_mean"]
    if clim_col in train_data:
        scores_clim = [f"{x}_hour" for x in scores] + [f"{x}_day" for x in scores]
        scores_skill = [f"skill_score_{x}_hour" for x in scores] + [
            f"skill_score_{x}_day" for x in scores
        ]
        scores += scores_clim
        scores += scores_skill
    df = pd.DataFrame.from_dict(
        {
            "train": [train_metrics[met] for met in scores],
            "valid": [valid_metrics[met] for met in scores],
            "test": [test_metrics[met] for met in scores],
        }
    )
    df["score_fn"] = scores
    df = df[["score_fn", "train", "valid", "test"]]
    df.to_csv(os.path.join(save_loc, "evaluate", "performance_metrics.csv"))

    # Save dfs
    keep_cols = total_input_vars + output_vars + ["y_pred", "Times"]
    if "latitide" not in train_data.columns:
        keep_cols += ["latitude", "longitude"]
    keep_cols = list(set(keep_cols))

    if "n_sites" in train_data:
        keep_cols.append("n_sites")
    if "n_site" in train_data:
        keep_cols.append("n_site")
    if "n_time" in train_data:
        keep_cols.append("n_time")
    for clim_col in climatologies:
        if clim_col in train_data:
            keep_cols.append(clim_col)

    train_data["y_pred"] = train_metrics["pred"].squeeze(-1)
    valid_data["y_pred"] = valid_metrics["pred"].squeeze(-1)
    test_data["y_pred"] = test_metrics["pred"].squeeze(-1)
    train_data[keep_cols].to_parquet(
        os.path.join(save_loc, "evaluate", "y_train.parquet")
    )
    valid_data[keep_cols].to_parquet(
        os.path.join(save_loc, "evaluate", "y_valid.parquet")
    )
    test_data[keep_cols].to_parquet(
        os.path.join(save_loc, "evaluate", "y_test.parquet")
    )
    
    train_data = pd.read_parquet(
        os.path.join(save_loc, "evaluate", "y_train.parquet"))
    valid_data = pd.read_parquet(
        os.path.join(save_loc, "evaluate", "y_valid.parquet"))
    test_data = pd.read_parquet(
        os.path.join(save_loc, "evaluate", "y_test.parquet"))

    # Make some figures
    figsize = (10, 7)
    fig, axs = plt.subplots(2, 3, figsize=figsize, sharex="col", sharey="row", frameon=False)
    names = ["Train", "Validation", "Test"]
    if "csv" in data_loc:
        bins = list(range(0, 300, 10))
        conus_max = 200
    else:
        bins = list(range(0, 30, 1))
        conus_max = 10
    splits = [train_metrics, valid_metrics, test_metrics]
    for k, (name, split) in enumerate(zip(names, splits)):
        ax1, ax2 = axs[0][k], axs[1][k]
        _ = ax1.hist2d(split["truth"].squeeze(-1), split["pred"].squeeze(-1), bins=bins)
        ax1.set_title(name)
        _ = ax2.hist(split["truth"], bins=bins, alpha=0.5, density=True)
        _ = ax2.hist(split["pred"], bins=bins, alpha=0.5, density=True)
        ax2.legend(["Truth", "Model"])
    axs[0][0].set_ylabel("True value")
    axs[1][0].set_xlabel("Predicted value")
    axs[1][1].set_xlabel("Predicted value")
    axs[1][2].set_xlabel("Predicted value")
    axs[1][0].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_loc, "evaluate", "result_histograms.png"), dpi=300, bbox_inches='tight')

    # CONUS
    splits = ["train", "valid", "test"]
    dfs = [train_data, valid_data, test_data]

    column = "10h_dead_fuel_moisture_content"  # dead
    if column not in train_data:
        column = "fuel_moisture_content"  # live

    for name, df in zip(names, dfs):
        df = drop_outliers(df, column, "y_pred")

        # Error -- RMSE, R2
        df_error = calc_site_r2_rmse(df, column, "y_pred")
        fn = os.path.join(save_loc, "evaluate", f"{name}_RMSE.png")
        plot_site_vars(df_error, "rmse", name, vmin=0, vmax=conus_max, fn=fn)
        fn = os.path.join(save_loc, "evaluate", f"{name}_R2.png")
        plot_site_vars(df_error, "r2", name, vmin=-1, vmax=1, cmap="coolwarm", fn=fn)

        if (
            column == "10h_dead_fuel_moisture_content"
        ):  # only have climatologies for dead for now
            skill_dfs = []
            for clim_col in climatologies:
                t = clim_col.split("-")[-1]
                dg = df[df[clim_col].notna()].copy()
                df_error_skill = calc_skill_scores(dg, column, "y_pred", clim_col)
                # RMSE skill score
                fn = os.path.join(save_loc, "evaluate", f"{name}_{t}_skill_rmse.png")
                plot_site_vars(
                    df_error_skill,
                    "rmse",
                    f"{t} climatology skill-score",
                    vmin=-1,
                    vmax=1,
                    cmap="coolwarm",
                    fn=fn,
                )
                # R2 skill score
                fn = os.path.join(save_loc, "evaluate", f"{name}_{t}_skill_r2.png")
                plot_site_vars(
                    df_error_skill,
                    "r2",
                    f"{t} climatology skill-score",
                    vmin=-1,
                    vmax=1,
                    cmap="coolwarm",
                    fn=fn,
                )

                # Create box-whisker for skill-score(metric) vs month
                monthly_skill = calc_skill_scores_site(dg, "y_pred", column, clim_col)
                skill_dfs.append(monthly_skill)
                
            figsize = (10, 3.5)
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, sharex="col", sharey="row", frameon=False
            )
            _ = sns.boxplot(x="month", y=f"rmse_skill_{climatologies[0]}", data=skill_dfs[0], ax=ax1)
            ax1.set_ylim([-1.0, 1.0])
            _ = sns.boxplot(
                x="month", y=f"rmse_skill_{climatologies[1]}", data=skill_dfs[1], ax=ax2
            ).set(ylabel=None)
            ax2.set_ylim([-1.0, 1.0])
            ax1.set_ylabel("Skill score (RMSE)")
            ax1.set_xlabel("Month")
            ax2.set_xlabel("Month")
            ax1.set_title("DOY-HR Climatology")
            ax2.set_title("DOY Climatology")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    save_loc, "evaluate", f"monthly_skill_{name}_rmse.png"
                ),
                dpi=300, bbox_inches='tight'
            )
            figsize = (10, 3.5)
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=figsize, sharex="col", sharey="row", frameon=False
            )
            _ = sns.boxplot(x="month", y=f"r2_skill_{climatologies[0]}", data=skill_dfs[0], ax=ax1)
            ax1.set_ylim([0.55, 1.02])
            _ = sns.boxplot(
                x="month", y=f"r2_skill_{climatologies[1]}", data=skill_dfs[1], ax=ax2
            ).set(ylabel=None)
            ax2.set_ylim([0.55, 1.02])
            ax1.set_ylabel(r"Skill score (R$^2$)")
            ax1.set_xlabel("Month")
            ax2.set_xlabel("Month")
            ax1.set_title("DOY-HR Climatology")
            ax2.set_title("DOY Climatology")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    save_loc, "evaluate", f"monthly_skill_{name}_r2_score.png"
                ),
                dpi=300, bbox_inches='tight', transparent = True
            )
            
    ### Plot the three together 
    df_train = drop_outliers(train_data, column, "y_pred")
    df_valid = drop_outliers(valid_data, column, "y_pred")
    df_test = drop_outliers(test_data, column, "y_pred")

    df_error_train = calc_site_r2_rmse(df_train, column, "y_pred")
    df_error_valid = calc_site_r2_rmse(df_valid, column, "y_pred")
    df_error_test = calc_site_r2_rmse(df_test, column, "y_pred")

    fn = os.path.join(save_loc, "evaluate", f"Total_RMSE.png")
    
    plot_site_vars(
        [df_error_train, df_error_valid, df_error_test], 
        "rmse", 
        None, 
        vmin=0, 
        vmax=conus_max, 
        fn=fn)

    fn = os.path.join(save_loc, "evaluate", f"Total_R2.png")
    plot_site_vars(
        [df_error_train, df_error_valid, df_error_test],
        "r2",
        None,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        fn=fn,
    )
    
    if column == "10h_dead_fuel_moisture_content":  
        # only have climatologies for dead for now
        for clim_col in climatologies:
            t = clim_col.split("-")[-1]
            dg_train = df_train[df_train[clim_col].notna()].copy()
            df_error_train = calc_skill_scores(dg_train, column, "y_pred", clim_col)
            dg_valid = df_valid[df_valid[clim_col].notna()].copy()
            df_error_valid = calc_skill_scores(dg_valid, column, "y_pred", clim_col)
            dg_test = df_test[df_test[clim_col].notna()].copy()
            df_error_test = calc_skill_scores(dg_test, column, "y_pred", clim_col)
            # RMSE skill score
            fn = os.path.join(save_loc, "evaluate", f"Total_{t}_skill_RMSE.png")
            plot_site_vars(
                [df_error_train, df_error_valid, df_error_test],
                "rmse",
                None,
                vmin=-1,
                vmax=1,
                cmap="coolwarm",
                fn=fn,
            )
            # R2 skill score
            fn = os.path.join(save_loc, "evaluate", f"Total_{t}_skill_R2.png")
            plot_site_vars(
                [df_error_train, df_error_valid, df_error_test],
                "r2",
                None,
                vmin=-1,
                vmax=1,
                cmap="coolwarm",
                fn=fn,
            )
    
    return


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py model.yml")
        sys.exit()

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    os.makedirs(os.path.join(save_loc, "evaluate"), exist_ok=True)

    # Evaluate
    evaluate(conf)
