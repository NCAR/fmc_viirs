import logging, sys, yaml, random, os, numpy as np
from fmcml.data import load_data, load_splitter, vegetatation_indices
from echo.src.base_objective import BaseObjective
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from collections import defaultdict
import xgboost as xgb
import pandas as pd
import tqdm, torch, shutil, joblib
import optuna

is_cuda = torch.cuda.is_available()
device = 0 if is_cuda else -1
print(f"Using CUDA: {is_cuda}")


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def trainer(conf, trial=False, verbose=True):
    seed = conf["seed"]
    seed_everything(seed)

    save_loc = conf["save_loc"]
    data_loc = conf["data_loc"]
    # input_vars = conf["input_vars"]
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

    if len(total_input_vars) == 0:
        if trial is not False:
            raise optuna.TrialPruned()
        else:
            raise OSError(
                "ECHO suggested no input columns, sigaling to prune this trial"
            )

    # static_vars = conf["static_vars"]
    output_vars = conf["output_vars"]
    verbose = conf["verbose"] if trial == False else False
    # total_input_vars = input_vars + static_vars
    splitter = conf["split_type"]

    # model config
    n_splits = conf["model"]["n_splits"]
    objective = conf["model"]["objective"]
    learning_rate = conf["model"]["learning_rate"]
    n_estimators = conf["model"]["n_estimators"]
    max_depth = conf["model"]["max_depth"]
    n_jobs = conf["model"]["n_jobs"]
    colsample_bytree = conf["model"]["colsample_bytree"]
    gamma = conf["model"]["gamma"]
    learning_rate = conf["model"]["learning_rate"]
    max_depth = conf["model"]["max_depth"]
    subsample = conf["model"]["subsample"]
    metric = conf["model"]["metric"]

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
        
    # Split the data
    data_folds = load_splitter(
        splitter, df, n_splits=n_splits, seed=seed, verbose=verbose
    )

    models = []
    results_dict = defaultdict(list)
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

        #         x_train = train_data[total_input_vars].to_numpy()
        #         x_valid = valid_data[total_input_vars].to_numpy()
        #         x_test  = test_data[total_input_vars].to_numpy()
        #         y_train = train_data[output_vars].to_numpy()
        #         y_valid = valid_data[output_vars].to_numpy()
        #         y_test  = test_data[output_vars].to_numpy()

        xgb_model = xgb.XGBRegressor(
            objective=objective,
            random_state=seed,
            gpu_id=device,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            subsample=subsample,
            n_jobs=n_jobs,
        )

        xgb_model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            early_stopping_rounds=10,
            verbose=verbose,
        )

        y_train = train_data[output_vars].to_numpy()
        y_valid = valid_data[output_vars].to_numpy()
        y_test  = test_data[output_vars].to_numpy()

        xgb_pred_train = scaler_y.inverse_transform(
            np.expand_dims(xgb_model.predict(x_train), 1)
        ).squeeze(-1)
        xgb_pred_valid = scaler_y.inverse_transform(
            np.expand_dims(xgb_model.predict(x_valid), 1)
        ).squeeze(-1)
        xgb_pred_test = scaler_y.inverse_transform(
            np.expand_dims(xgb_model.predict(x_test), 1)
        ).squeeze(-1)

        results_dict["train_rmse"].append(
            mean_squared_error(y_train, xgb_pred_train) ** (1 / 2)
        )
        results_dict["valid_rmse"].append(
            mean_squared_error(y_valid, xgb_pred_valid) ** (1 / 2)
        )
        results_dict["test_rmse"].append(
            mean_squared_error(y_test, xgb_pred_test) ** (1 / 2)
        )
        results_dict["train_mae"].append(mean_absolute_error(y_train, xgb_pred_train))
        results_dict["train_r2"].append(r2_score(y_train, xgb_pred_train))
        results_dict["valid_mae"].append(mean_absolute_error(y_valid, xgb_pred_valid))
        results_dict["valid_r2"].append(r2_score(y_valid, xgb_pred_valid))
        results_dict["test_mae"].append(mean_absolute_error(y_test, xgb_pred_test))
        results_dict["test_r2"].append(r2_score(y_test, xgb_pred_test))
        results_dict["fold"].append(k_fold)
        models.append([scaler_x, scaler_y, xgb_model])

    fold_results = pd.DataFrame.from_dict(results_dict).reset_index()
    best_fold = [
        i for i, j in enumerate(fold_results[metric]) if j == min(fold_results[metric])
    ][0]

    if trial == False:
        logging.info(f"The best fold with metric {metric} was {best_fold}")
        return models[best_fold], fold_results, best_fold

    results = {
        #"fold": best_fold,
        "train_rmse": np.mean(fold_results["train_rmse"]),
        "train_rmse_std": np.std(fold_results["train_rmse"]),
        "valid_rmse": np.mean(fold_results["valid_rmse"]),
        "valid_rmse_std": np.std(fold_results["valid_rmse"]),
        "test_rmse": np.mean(fold_results["test_rmse"]),
        "test_rmse_std": np.std(fold_results["test_rmse"]),
        "train_r2": np.mean(fold_results["train_r2"]),
        "train_r2_std": np.std(fold_results["train_r2"]),
        "valid_r2": np.mean(fold_results["valid_r2"]),
        "valid_r2_std": np.std(fold_results["valid_r2"]),
        "test_r2": np.mean(fold_results["test_r2"]),
        "test_r2_std": np.std(fold_results["test_r2"]),
    }

    return results


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        return trainer(conf, trial=trial, verbose=False)


#         try:
#             return trainer(conf, trial = trial, verbose = False)
#         except Exception as E:
#             if "CUDA" in str(E):
#                 logging.warning(
#                     f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}.")
#                 raise optuna.TrialPruned()
#             else:
#                 logging.warning(
#                     f"Trial {trial.number} failed due to error: {str(E)}.")
#                 raise E


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python train_mlp.py model.yml")
        sys.exit()

    # ### Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # ### Load the configuration and get the relevant variables
    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)
    if not os.path.join(save_loc, "model.yml"):
        shutil.copyfile(config, os.path.join(save_loc, "model.yml"))

    model, results, best_fold = trainer(conf)
    scaler_x, scaler_y, model = model 
    print(results)

    with open(os.path.join(save_loc, f"model_results.pkl"), "wb") as fid:
        joblib.dump([model, results], fid)
        
    with open(os.path.join(save_loc, "scalers.pkl"), "wb") as fid:
        joblib.dump([scaler_x, scaler_y, best_fold], fid)
        
    results.to_csv(os.path.join(save_loc, "training_metrics_ensemble.csv"))
    mean = results.mean(axis=0)
    std = results.std(axis=0)
    pd.concat([mean, std], axis = 1).rename(
        {"0": "mean", "1": "std"}
    ).to_csv(os.path.join(save_loc, "training_metrics_averages.csv"))
