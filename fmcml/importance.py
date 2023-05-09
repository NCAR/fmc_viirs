from fmcml.model import get_device
from captum import attr
import functools as ft
import pandas as pd
import numpy as np
import logging
import random
import torch
import tqdm
import shap
import gc


def mlp_importance(model, 
                   x_train, 
                   x_test, 
                   y_test, 
                   loss, 
                   input_vars, 
                   n_repeats=30, 
                   batch_size=128, 
                   shuffle=False, 
                   batches=1e10):
    
    # permutation
    logging.info("Computing permutation importances")
    feat_imp_diz = permut_features_importance(
        model, x_test, y_test.squeeze(-1), 
        loss_fun=loss, 
        n_repeats=n_repeats, 
        columns_name=input_vars)

    feat_imp_mean, feat_imp_std = feat_imp_diz
    feat_imp_mean = pd.Series(feat_imp_mean)
    feat_imp_std = pd.Series(feat_imp_std)
    
    # shap, deep lift, etc
    logging.info("Computing SHAP, IG (and others) importances")
    ig_attr_test = []
    ig_nt_attr_test = []
    dl_attr_test = []
    gs_attr_test = []
    fa_attr_test = []

    ig = attr.IntegratedGradients(model)
    ig_nt = attr.NoiseTunnel(ig)
    dl = attr.DeepLift(model)
    gs = attr.GradientShap(model)
    fa = attr.FeatureAblation(model)

    device = get_device()
    indices = list(range(x_test.shape[0]))
    for k in tqdm.tqdm(range(len(indices) // batch_size)):
        if random:
            random.shuffle(indices)
            _x = torch.from_numpy(
                x_test[indices[:batch_size]]
            ).to(device).float()
        else:
            _x = torch.from_numpy(
                x_test[k * batch_size: (k + 1) * batch_size]
            ).to(device).float()

        ig_attr_test.append(ig.attribute(_x, n_steps=50).cpu())
        ig_nt_attr_test.append(ig_nt.attribute(_x).cpu())
        gc.collect()

        dl_attr_test.append(dl.attribute(_x).cpu())
        gc.collect()

        _x_comp = torch.from_numpy(x_train).to(device).float()
        gs_attr_test.append(gs.attribute(_x, _x_comp).cpu())
        del _x_comp
        gc.collect()

        fa_attr_test.append(fa.attribute(_x).cpu())
        gc.collect()
        
        if (k + 1) >= batches:
            break
        
    ig_attr_test = torch.vstack(ig_attr_test)
    ig_nt_attr_test = torch.vstack(ig_nt_attr_test)
    dl_attr_test = torch.vstack(dl_attr_test)
    gs_attr_test = torch.vstack(gs_attr_test)
    fa_attr_test = torch.vstack(fa_attr_test)
    
    ig_attr_test_sum = ig_attr_test.detach().numpy().sum(0)
    ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

    ig_nt_attr_test_sum = ig_nt_attr_test.detach().numpy().sum(0)
    ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)

    dl_attr_test_sum = dl_attr_test.detach().numpy().sum(0)
    dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

    gs_attr_test_sum = gs_attr_test.detach().numpy().sum(0)
    gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

    fa_attr_test_sum = fa_attr_test.detach().numpy().sum(0)
    fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

    lin_weight = model.fcn[0].weight[0].detach().cpu().numpy()
    y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)
    
    importance_results = pd.DataFrame.from_dict({
        "feature": input_vars,
        "permutation": feat_imp_mean / feat_imp_mean.sum(),
        "integrated_gradients": ig_attr_test_norm_sum,
        "noise_tunnel": ig_nt_attr_test_norm_sum,
        "deep_lift": dl_attr_test_norm_sum,
        "gradient_shap": gs_attr_test_norm_sum,
        "feature_ablation": fa_attr_test_norm_sum,
        "layer1_weights": y_axis_lin_weight
    })
    sort_cols = [
        "permutation", "integrated_gradients", "noise_tunnel",
        "deep_lift", "gradient_shap", "feature_ablation", "layer1_weights"
    ]
    new_order = importance_results[sort_cols].abs().mean(axis=1).sort_values(ascending=False).index
    importance_results = importance_results.reindex(new_order)
    importance_results = importance_results.reset_index(drop = True)
    return importance_results


def xgb_importance(model, 
                   x_train, 
                   x_test, 
                   y_test, 
                   loss, 
                   input_vars, 
                   n_repeats=30, 
                   batch_size=128, 
                   shuffle=False, 
                   batches=1e10):
    
    ### Gain
    gain_importance = model.get_booster().get_score(importance_type='gain')
    sorted_gain = sorted([x for x in gain_importance.items()], key = lambda x: x[1], reverse = True)
    ylabels = [input_vars[int(x[0].strip("f"))] for x in sorted_gain][::-1]
    gain_importance = pd.DataFrame.from_dict({
        "feature": ylabels[::-1],
        "gain": [_[1] for _ in sorted_gain]
    })
    
    ### Permutation
    xgb_feat_imp_diz = permut_features_importance(
        model, x_test, y_test.squeeze(-1), 
        loss_fun=loss, 
        n_repeats=n_repeats, 
        columns_name=input_vars
    )
    xgb_feat_imp_mean, xgb_feat_imp_std = xgb_feat_imp_diz
    xgb_feat_imp_mean = pd.Series(xgb_feat_imp_mean)
    xgb_feat_imp_std = pd.Series(xgb_feat_imp_std)
    perm_importance = pd.DataFrame.from_dict({
        "feature": input_vars,
        "permutation": xgb_feat_imp_mean
    })
    
    ### Shap
    explainer = shap.TreeExplainer(model, x_train)
    if batches < (x_test.shape[0] // batch_size):
        xgb_shap_values = explainer.shap_values(x_test[:(batches * batch_size)])
    else:
        xgb_shap_values = explainer.shap_values(x_test)
    xgb_shap_mean = pd.Series(np.abs(xgb_shap_values).mean(axis=0))
    xgb_shap_std = pd.Series(np.abs(xgb_shap_values).std(axis=0))
    shap_importance = pd.DataFrame.from_dict({
        "feature": input_vars,
        "shap": xgb_shap_mean
    })
    
    ### Merge
    dfs = [gain_importance, perm_importance, shap_importance]
    xgb_importance = ft.reduce(lambda left, right: pd.merge(left, right, on='feature'), dfs) 
    metrics = ["gain", "permutation", "shap"]
    xgb_importance["gain"] /= xgb_importance["gain"].sum()
    xgb_importance["permutation"] /= xgb_importance["permutation"].sum()
    xgb_importance["shap"] /= xgb_importance["shap"].sum()
    xgb_new_order = xgb_importance[metrics].abs().mean(axis=1).sort_values(ascending=False).index
    xgb_importance = xgb_importance.reindex(xgb_new_order)
    return xgb_importance, metrics


def permut_features_importance(
    model, 
    X, 
    y, 
    loss_fun, 
    n_repeats=10,
    columns_name=None
):
    
    X = np.asarray(X)
    assert X.ndim == 2
    
    y = np.asarray(y)
    assert y.ndim < 2
    
    if columns_name is not None:
        assert len(columns_name) == X.shape[-1]
    else:
        columns_name = np.arange(X.shape[-1])
    
    yp = model.predict(X)
    error = loss_fun(yp, y)
    features_imp = {}
    std_features_imp = {}     
        
    for col in tqdm.tqdm(range(X.shape[-1])):
        _importance = []
        for _ in range(n_repeats):
            _X = np.copy(X)
            _X[:,col] = np.random.permutation(_X[:,col])
            yp = model.predict(_X)
            _importance.append(loss_fun(yp, y) - error)
                
        features_imp[columns_name[col]] = np.mean(_importance)
        std_features_imp[columns_name[col]] = np.std(_importance)
                
    return features_imp, std_features_imp


def variable_importance(data, 
                        labels, 
                        variable_names, 
                        model_name, 
                        model, 
                        score_func, 
                        permutations=30,
                        sklearn_model=False,
                        mean_model=False):
    
    if sklearn_model:
        preds = model.predict_proba(data)[:, 1]
    else:
        preds = model.predict(data)[:, 0]
    score = score_func(labels, preds)
    indices = np.arange(preds.shape[0])
    perm_data = np.copy(data)
    var_scores = pd.DataFrame(index=np.arange(permutations + 1), columns=variable_names, dtype=float)
    var_scores.loc[0, :] = score
    data_shape_len = len(data.shape)
    print(data.shape)
    for v, variable in tqdm.tqdm(enumerate(variable_names)):
        for p in range(1, permutations + 1):
            np.random.shuffle(indices)
            if mean_model and data_shape_len == 2:
                perm_data[:, v] = data[indices, v]
            elif mean_model and data_shape_len == 3:
                perm_data[:, :, v] = data[indices, :, v]
            else:
                perm_data[:, :, :, v] = data[indices, :, :, v]
            if sklearn_model:
                perm_preds = model.predict_proba(perm_data)[:, 1]
            else:
                perm_preds = model.predict(perm_data)[:, 0]
            var_scores.loc[p, variable] = score_func(labels, perm_preds)
        if mean_model:
            perm_data[:, v] = data[:, v]
        elif mean_model and data_shape_len == 3:
            perm_data[:, :, v] = data[:, :, v]
        else:
            perm_data[:, :, :, v] = data[:, :, :, v]
        score_diff = (var_scores.loc[0, variable] - var_scores.loc[1:, variable]) /  var_scores.loc[0, variable]
        #print(model_name, variable, score_diff.mean(), score_diff.std())
    return var_scores


def variable_importance_faster(data, 
                               labels, 
                               variable_names, 
                               model_name, 
                               model, 
                               score_funcs, 
                               permutations=30,
                               sklearn_model=False,
                               mean_model=False):
    
    if sklearn_model:
        preds = model.predict_proba(data)[:, 1]
    else:
        preds = model.predict(data)[:, 0]
    scores = [sf(labels, preds) for sf in score_funcs]
    indices = np.arange(preds.shape[0])
    perm_data = np.copy(data)
    var_scores = []
    for s in range(len(score_funcs)):
        var_scores.append(pd.DataFrame(index=np.arange(permutations + 1), columns=variable_names, dtype=float))
        var_scores[-1].loc[0, :] = scores[s]
    data_shape_len = len(data.shape)
    for p in tqdm.tqdm(range(1, permutations + 1)):
        np.random.shuffle(indices)
        for v, variable in enumerate(variable_names):
            if mean_model and data_shape_len == 2:
                perm_data[:, v] = data[indices, v]
            elif mean_model and data_shape_len == 3:
                perm_data[:, :, v] = data[indices, :, v]
            else:
                perm_data[:, :, :, v] = data[indices, :, :, v]
            if sklearn_model:
                perm_preds = model.predict_proba(perm_data)[:, 1]
            else:
                perm_preds = model.predict(perm_data)[:, 0]
            for s, score_func in enumerate(score_funcs):
                var_scores[s].loc[p, variable] = score_func(labels, perm_preds)
            if mean_model:
                perm_data[:, v] = data[:, v]
            elif mean_model and data_shape_len == 3:
                perm_data[:, :, v] = data[:, :, v]
            else:
                perm_data[:, :, :, v] = data[:, :, :, v]
            #print(model_name, variable)
    return var_scores