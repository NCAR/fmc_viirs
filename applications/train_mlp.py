import logging, sys, yaml, random, os, torch, numpy as np
from fmcml.data import load_data, load_splitter, vegetatation_indices
from fmcml.model import init_weights, DNN
from fmcml.loss import load_loss
from echo.src.base_objective import BaseObjective
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from torch.utils.data.dataset import TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
import pandas as pd
import tqdm, optuna, shutil, os
import joblib

is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
print(f"Using CUDA: {is_cuda}")


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    

def trainer(conf, trial = False, verbose = True):
    seed = conf["seed"]
    seed_everything(seed)
    
    save_loc = conf["save_loc"]
    data_loc = conf["data_loc"]
    
    #input_vars = conf["input_vars"]
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
                "ECHO suggested no input columns, sigaling to prune this trial")

    output_vars = conf["output_vars"]
    verbose = conf["verbose"]
    #total_input_vars = input_vars + static_vars
    splitter = conf["split_type"]
    
    # model config
    input_size = len(total_input_vars)
    middle_size = conf["model"]["middle_size"] 
    output_size = len(output_vars)
    dropout = conf["model"]["dropout"]
    num_layers = conf["model"]["num_layers"]
    #optimzer config
    learning_rate = conf["optimizer"]["learning_rate"]
    L2_penalty = conf["optimizer"]["L2_penalty"]
    # trainer config
    n_splits = conf["trainer"]["n_splits"]
    batch_size = conf["trainer"]["batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]
    lr_patience = conf["trainer"]["lr_patience"]
    stopping_patience = conf["trainer"]["stopping_patience"]
    epochs = conf["trainer"]["epochs"]
    batches_per_epoch = conf["trainer"]["batches_per_epoch"]
    training_loss = conf["trainer"]["training_loss"]
    metric = conf["trainer"]["metric"]
    
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
        
    else:
        df = load_data(
            data_loc,
            verbose = verbose,
            filter_input_vars = total_input_vars, 
            filter_output_vars = output_vars
        )
    
    # Split the data
    data_folds = load_splitter(
        splitter,
        df, 
        n_splits = n_splits, 
        seed = seed,
        verbose = verbose
    )
    
    fold_results = []
    fold_dict = defaultdict(list)
    for k_fold, (train_data, valid_data, test_data) in enumerate(data_folds):
        scaler_x = QuantileTransformer(n_quantiles=1000, random_state=seed, output_distribution = "normal") #StandardScaler()
        scaler_y = QuantileTransformer(n_quantiles=1000, random_state=seed, output_distribution = "normal") #StandardScaler()
        x_train = scaler_x.fit_transform(train_data[total_input_vars])
        x_valid = scaler_x.transform(valid_data[total_input_vars])
        x_test  = scaler_x.transform(test_data[total_input_vars])
        y_train = scaler_y.fit_transform(train_data[output_vars])
        y_valid = scaler_y.transform(valid_data[output_vars])
        y_test  = scaler_y.transform(test_data[output_vars])
        
        train_split = TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).float()
        )
        train_loader = DataLoader(train_split, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=8)

        valid_split = TensorDataset(
            torch.from_numpy(x_valid).float(),
            torch.from_numpy(y_valid).float()
        )
        valid_loader = DataLoader(valid_split, 
                                  batch_size=valid_batch_size, 
                                  shuffle=False, 
                                  num_workers=0)
        test_split = TensorDataset(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_test).float()
        )
        test_loader = DataLoader(test_split, 
                                  batch_size=valid_batch_size, 
                                  shuffle=False, 
                                  num_workers=0)
        
        ### Load model 
        model = DNN(
            input_size, 
            output_size, 
            block_sizes = [middle_size for k in range(num_layers)],
            dr = [dropout for k in range(num_layers)]
        )
        #init_weights(model, init_type = "kaiming", verbose = verbose)
        model = model.to(device)
        
        ### Load optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = learning_rate,
            weight_decay = L2_penalty
        )
        
        ### Load losses
        train_loss_fn = load_loss(training_loss) #torch.nn.SmoothL1Loss()
        valid_loss_fn = torch.nn.L1Loss()
        test_loss_fn = torch.nn.L1Loss()
        
        ### Load schedulers 
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience = lr_patience, 
            verbose = verbose,
            min_lr = 1.0e-13
        )
        
        ### TRAIN AND VALIDATE 
        results_dict = defaultdict(list)
    
        my_iter = list(range(epochs))

        for epoch in my_iter:

            # Train in batch mode
            model.train()

            train_loss = []
            train_iter = tqdm.tqdm(enumerate(train_loader),
                                total = min(batches_per_epoch, x_train.shape[0] // batch_size),
                                leave = True)

            for k, (x, y) in train_iter:
                optimizer.zero_grad()
                loss = train_loss_fn(y.to(device), model(x.to(device)))
                train_loss.append(loss.item())

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                if not np.isfinite(loss.item()):
                    if trial is not False:
                        raise optuna.TrialPruned()
                    else:
                        raise OSError(f"Loss {training_loss} returned nan")

                print_str = f'Epoch {epoch} train_loss {np.mean(train_loss):4f}'
                if verbose:
                    train_iter.set_description(print_str)
                    train_iter.refresh()

                if k % batches_per_epoch == 0 and k > 0:
                    break
            results_dict["train_loss"].append(np.mean(train_loss))

            # Validate 
            model.eval()
            with torch.no_grad():

                # Validate in batch mode
                valid_loss, valid_rmse, valid_r2 = [], [], []
                valid_iter = tqdm.tqdm(enumerate(valid_loader),
                                       total = x_valid.shape[0] // valid_batch_size,
                                       leave = True)
                
                for k, (x, y) in valid_iter:
                    pred = model(x.to(device))
                    loss = valid_loss_fn(y.to(device), pred)
                    valid_loss.append(loss.item())
                    y_inv = scaler_y.inverse_transform(y).squeeze(-1)
                    y_pred_inv = scaler_y.inverse_transform(pred.cpu()).squeeze(-1)
                    valid_rmse.append(mean_squared_error(y_inv, y_pred_inv)**(1/2))
                    valid_r2.append(r2_score(y_inv, y_pred_inv))
                    print_str = f'Epoch {epoch} valid_loss {np.mean(valid_loss):4f} valid_rmse: {np.mean(valid_rmse):.4f} valid_r2: {np.mean(valid_r2):.4f}'
                    if verbose:
                        valid_iter.set_description(print_str)
                        valid_iter.refresh()
                    
            if not np.isfinite(np.mean(valid_loss)):
                break

            results_dict["epoch"].append(epoch)
            #results_dict["train_loss"].append(np.mean(train_loss))
            results_dict["valid_mae"].append(np.mean(valid_loss))
            results_dict["valid_rmse"].append(np.mean(valid_rmse))
            results_dict["valid_r2"].append(np.mean(valid_r2))
            results_dict["lr"].append(optimizer.param_groups[0]['lr'])

            # Save the dataframe to disk
            df = pd.DataFrame.from_dict(results_dict).reset_index()
            if verbose:
                df.to_csv(f"{save_loc}/training_log.csv", index = False)

            # anneal the learning rate using just the box metric
            lr_scheduler.step(results_dict[metric][-1])

            if results_dict[metric][-1] == min(results_dict[metric]):
                state_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': min(results_dict[metric])
                }
                torch.save(state_dict, f"{save_loc}/mlp.pt")

            # Stop training if we have not improved after X epochs
            best_epoch = [i for i,j in enumerate(results_dict[metric]) if j == min(results_dict[metric])][0]
            offset = epoch - best_epoch
            if offset >= stopping_patience:
                break
                
        # Select the best model 
        checkpoint = torch.load(
            f"{save_loc}/mlp.pt", map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Predict on the three splits and compute metrics 
        model.eval()
        with torch.no_grad():
            X = np.array_split(x_train, x_train.shape[0] / valid_batch_size)
            pred = scaler_y.inverse_transform(torch.cat(
                [model(torch.from_numpy(_x).to(device).float()).cpu() for _x in X]
            ))
            true = train_data[output_vars].to_numpy()
            fold_dict["train_rmse"].append(mean_squared_error(true, pred)**(1/2))
            fold_dict["train_mae"].append(mean_absolute_error(true, pred))
            fold_dict["train_r2"].append(r2_score(true, pred))
            # valid
            X = np.array_split(x_valid, x_valid.shape[0] / valid_batch_size)
            pred = scaler_y.inverse_transform(torch.cat(
                [model(torch.from_numpy(_x).to(device).float()).cpu() for _x in X]
            ))
            true = valid_data[output_vars].to_numpy()
            fold_dict["valid_rmse"].append(mean_squared_error(true, pred)**(1/2))
            fold_dict["valid_mae"].append(mean_absolute_error(true, pred))
            fold_dict["valid_r2"].append(r2_score(true, pred))
            # test
            X = np.array_split(x_test, x_test.shape[0] / valid_batch_size)
            pred = scaler_y.inverse_transform(torch.cat(
                [model(torch.from_numpy(_x).to(device).float()).cpu() for _x in X]
            ))
            true = test_data[output_vars].to_numpy()
            fold_dict["test_rmse"].append(mean_squared_error(true, pred)**(1/2))
            fold_dict["test_mae"].append(mean_absolute_error(true, pred))
            fold_dict["test_r2"].append(r2_score(true, pred))
            fold_dict["fold"].append(k_fold)
        
        fold_results = pd.DataFrame.from_dict(fold_dict).reset_index()
        best_fold = [
            i for i, j in enumerate(fold_results[metric]) if j == min(fold_results[metric])
        ][0]
        
        if best_fold == k_fold:
            shutil.copy(f"{save_loc}/mlp.pt", f"{save_loc}/best.pt")
            shutil.copy(f"{save_loc}/training_log.csv", f"{save_loc}/best_training_log.csv.pt")
            with open(f"{save_loc}/scalers.pkl", "wb") as fid:
                joblib.dump([scaler_x, scaler_y, best_fold], fid)
        
    fold_results = pd.DataFrame.from_dict(fold_dict).reset_index()
        
    if trial == False:
        return fold_results
        
    best_fold = [i for i,j in enumerate(fold_results[metric]) if j == min(fold_results[metric])][0]
    results = {
        "split": best_fold,
        "train_loss": fold_results["train_loss"][best_fold],
        "valid_mae": fold_results["valid_mae"][best_fold],
        "valid_rmse": fold_results["valid_rmse"][best_fold],
        "valid_r2": fold_results["valid_r2"][best_fold]
    }
    return results


class Objective(BaseObjective):

    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        try:
            return trainer(conf, trial = trial, verbose = False)
        except Exception as E:
            if "CUDA" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}.")
                raise optuna.TrialPruned()
            elif "reraise" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to unspecified error: {str(E)}.")
                raise optuna.TrialPruned()
            else:
                logging.warning(
                    f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E
        

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python train_mlp.py model.yml")
        sys.exit()
        
    # ### Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

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
    os.makedirs(save_loc, exist_ok = True)
    if not os.path.join(save_loc, "model.yml"):
        shutil.copyfile(config, os.path.join(save_loc, "model.yml"))
        
    results = trainer(conf)
    print(results)
    
    results.to_csv(os.path.join(save_loc, "training_metrics_ensemble.csv"))
    mean = results.mean(axis=0)
    std = results.std(axis=0)
    pd.concat([mean, std], axis = 1).rename(
        {"0": "mean", "1": "std"}
    ).to_csv(os.path.join(save_loc, "training_metrics_averages.csv"))
        
