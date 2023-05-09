from sklearn.preprocessing import StandardScaler, QuantileTransformer
import logging
import sys
import yaml
import torch

import pandas as pd
import os
from fmcml.importance import mlp_importance, xgb_importance
from fmcml.model import DNN, get_device
from fmcml.seed import seed_everything
from fmcml.data import load_data, load_splitter, vegetatation_indices
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def run(conf, n_repeats=30, batch_size=128, shuffle=False, batches=1e10):

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

    if "n_splits" in conf["model"]:
        model_type = "xgb"
        n_splits = conf["model"]["n_splits"]
    else:
        model_type = "mlp"
        n_splits = conf["trainer"]["n_splits"]

    # Load the data and take the first split
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

    else:
        df = load_data(
            data_loc,
            verbose=verbose,
            filter_input_vars=total_input_vars,
            filter_output_vars=output_vars,
        )

    data_folds = load_splitter(
        conf["split_type"], df, n_splits=n_splits, seed=seed, verbose=conf["verbose"]
    )

    for k_fold, (train_data, valid_data, test_data) in enumerate(data_folds):
        scaler_x = QuantileTransformer(
            n_quantiles=1000, random_state=seed, output_distribution="normal"
        )  # StandardScaler()
        scaler_y = QuantileTransformer(
            n_quantiles=1000, random_state=seed, output_distribution="normal"
        )  # StandardScaler()
        x_train = scaler_x.fit_transform(train_data[total_input_vars])
        x_valid = scaler_x.transform(valid_data[total_input_vars])
        x_test = scaler_x.transform(test_data[total_input_vars])
        y_train = scaler_y.fit_transform(train_data[output_vars])
        y_valid = scaler_y.transform(valid_data[output_vars])
        y_test = scaler_y.transform(test_data[output_vars])
        break

    """
        MLP importance
    """

    if model_type == "mlp":

        # Load the model and weights
        input_size = len(total_input_vars)
        middle_size = conf["model"]["middle_size"]
        output_size = len(output_vars)
        dropout = conf["model"]["dropout"]
        num_layers = conf["model"]["num_layers"]

        device = get_device()
        model = DNN(
            input_size,
            output_size,
            block_sizes=[middle_size for k in range(num_layers)],
            dr=[dropout for k in range(num_layers)],
        ).to(device)

        checkpoint = torch.load(
            f"{save_loc}/mlp.pt", map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Compute importances
        results = mlp_importance(
            model,
            x_train,
            x_test,
            y_test,
            mean_squared_error,
            total_input_vars,
            n_repeats=n_repeats,
            batch_size=batch_size,
            shuffle=shuffle,
            batches=batches,
        )
        results.to_csv(os.path.join(save_loc, "evaluate", "importance.csv"))

        # Plot the results
        results = pd.read_csv(os.path.join(save_loc, "evaluate", "importance.csv"))
        results.index = list(results["feature"])
        results = results.drop(columns=["feature"])
        if "Unnamed: 0" in results.columns:
            results = results.drop(columns=["Unnamed: 0"])

        sns.set_style("dark")
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, figsize=(15, 10), sharex=True, sharey="row"
        )

        ax1 = results.plot(kind="bar", ax=ax1, stacked=True, legend=False, fontsize=10)
        ax1.grid()

        ax2 = results.abs().plot(
            kind="bar", ax=ax2, stacked=True, legend=False, fontsize=10
        )
        ax2.set_xticks(range(results.shape[0]))
        _ = ax2.set_xticklabels(
            list(results.index),
            rotation=45,
            rotation_mode="anchor",
            ha="right",
            fontsize=10,
        )
        ax2.grid()

        ax1.legend(results.columns, ncol=2, fontsize=10, loc="best")
        ax1.set_ylabel("Normalized importance", fontsize=10)
        ax2.set_ylabel("Absolute normalized importance", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, "evaluate", "importance.pdf"), dpi=300)

    """
        XGB
    """

    if model_type == "xgb":

        with open(os.path.join(conf["save_loc"], "model_results.pkl"), "rb") as fid:
            model, _ = joblib.load(fid)

        results, metrics = xgb_importance(
            model,
            x_train,
            x_test,
            y_test,
            mean_squared_error,
            total_input_vars,
            n_repeats=n_repeats,
            batch_size=batch_size,
            shuffle=shuffle,
            batches=batches,
        )
        results.to_csv(os.path.join(save_loc, "evaluate", "importance.csv"))

        metrics = ["gain", "permutation", "shap"]
        # results = pd.read_csv(os.path.join(save_loc,  "evaluate", "importance.csv"))

        sns.set_style("dark")
        fig, ax1 = plt.subplots(
            nrows=1, ncols=1, figsize=(10, 5.5), sharex=True, sharey="row"
        )
        ax1 = (
            results[metrics]
            .abs()
            .plot(kind="bar", ax=ax1, stacked=True, legend=False, fontsize=10)
        )
        ax1.set_xticks(range(results.shape[0]))
        tick_labels = results["feature"]
        tick_labels = [
            x.replace("_medium", "").replace("750m Surface Reflectance Band", "Sfc rfl")
            for x in tick_labels
        ]
        tick_labels = [
            x.replace("750m Surface Reflectance Band", "Sfc rfl") for x in tick_labels
        ]
        tick_labels = [
            x.replace("375m Surface Reflectance Band", "Sfc rfl") for x in tick_labels
        ]
        tick_labels = [x.replace("_", " ") for x in tick_labels]
        _ = ax1.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=12)
        ax1.grid()

        ax1.legend(metrics, ncol=1, fontsize=12, loc="best")
        ax1.set_ylabel("Absolute normalized importance", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_loc, "evaluate", "importance.png"), dpi=300)

    return results


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python feature_importance.py model.yml")
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

    results = run(conf)
