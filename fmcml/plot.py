import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from collections import defaultdict
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def plot_site_vars(df, var_name, title_pre, vmin=0, vmax=50, cmap=None, fn=None):
    
    if cmap == None:
        cmap = plt.get_cmap('gnuplot', 10)
    
    state_borders = gpd.read_file(
        "/glade/work/schreck/repos/fmc-repo/fmc/notebooks/data/usa-states-census-2014.shp"
    )
    state_borders = state_borders.to_crs("EPSG:3395")
    symbols = ["o", "s", "^"]
    if type(df) == list:
        fig, ax = plt.subplots(figsize=(7, 5), frameon=False)
        for i, dff in enumerate(df):
            var_gdf = gpd.GeoDataFrame(
                dff[var_name], geometry=gpd.points_from_xy(dff["longitude"], dff["latitude"])
            )
            var_gdf = var_gdf.set_crs("epsg:4326")
            var_gdf = var_gdf.to_crs(state_borders.crs)
            state_borders.boundary.plot(ax=ax, color="black")
            var_gdf.plot(
                column=var_name,
                ax=ax,
                cmap=cmap,
                legend=True if i == len(symbols)-1 else False,
                legend_kwds={"shrink": 0.3},
                marker=symbols[i],
                markersize=10,
                vmin=vmin,
                vmax=vmax
            )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(fn, dpi=300, transparent = True, bbox_inches='tight')
    
    else:
        var_gdf = gpd.GeoDataFrame(
            df[var_name], geometry=gpd.points_from_xy(df["longitude"], df["latitude"])
        )
        var_gdf = var_gdf.set_crs("epsg:4326")
        var_gdf = var_gdf.to_crs(state_borders.crs)
        fig, ax = plt.subplots(figsize=(7, 5), frameon=False)
        state_borders.boundary.plot(ax=ax, color="black")
        var_gdf.plot(
            column=var_name,
            ax=ax,
            cmap=cmap,
            legend=True,
            legend_kwds={"shrink": 0.3},
            marker="^",
            markersize=10,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{title_pre} {var_name.upper()}")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(fn, dpi=300, transparent = True, bbox_inches='tight')
        return

def drop_chebychev(df, col, k=6):
    mu = df[col].mean()
    sigma = df[col].std()
    ubound = mu + k * sigma
    lbound = mu - k * sigma
    drop_idxs = df[(df[col] < lbound) | (df[col] > ubound)].index
    return df.drop(drop_idxs)


def drop_outliers(df, y_col, y_hat_col):
    df = drop_chebychev(df, y_col)
    df = drop_chebychev(df, y_hat_col)
    return df


def calc_site_r2_rmse(df, y_col, y_hat_col):
    def r2_rmse(grp):
        r2 = r2_score(grp[y_col], grp[y_hat_col])
        rmse = np.sqrt(mean_squared_error(grp[y_col], grp[y_hat_col]))
        return pd.Series(dict(r2=r2, rmse=rmse))

    try:
        df_error = (
            df.groupby(["n_site", "latitude", "longitude"]).apply(r2_rmse).reset_index()
        )
    except:
        df_error = (
            df.groupby(["n_sites", "latitude", "longitude"])
            .apply(r2_rmse)
            .reset_index()
        )

    # Cap r2 at -1,1
    df_error.loc[df_error["r2"] < -1.0, "r2"] = -1.0
    df_error.loc[df_error["r2"] > 1.0, "r2"] = 1.0
    # print(df_error['rmse'].max())
    return df_error


def calc_skill_scores(df, y_col, y_hat_col, clim_col):
    def r2_rmse(grp):
        r2_ML = r2_score(grp[y_col], grp[y_hat_col])
        rmse_ML = np.sqrt(mean_squared_error(grp[y_col], grp[y_hat_col]))
        r2_c = r2_score(grp[y_col], grp[clim_col])
        rmse_c = np.sqrt(mean_squared_error(grp[y_col], grp[clim_col]))

        skill_rmse = 1 - rmse_ML / rmse_c
        skill_r2 = (r2_ML - r2_c) / (1 - r2_c)

        return pd.Series(dict(r2=skill_r2, rmse=skill_rmse))

    try:
        df_error = (
            df.groupby(["n_site", "latitude", "longitude"]).apply(r2_rmse).reset_index()
        )
    except:
        df_error = (
            df.groupby(["n_sites", "latitude", "longitude"])
            .apply(r2_rmse)
            .reset_index()
        )

    # Cap r2 at -1,1
    df_error.loc[df_error["r2"] < -1.0, "r2"] = -1.0
    df_error.loc[df_error["r2"] > 1.0, "r2"] = 1.0
    # print(df_error['rmse'].max())
    return df_error


def calc_skill_scores_site(dataframe, y_col, y_hat_col, clim_col):
    dataframe["month"] = pd.to_datetime(dataframe["n_time"]).dt.month
    dataframe["day"] = pd.to_datetime(dataframe["n_time"]).dt.dayofyear
    dataframe["hour"] = pd.to_datetime(dataframe["n_time"]).dt.hour
    results_dict = defaultdict(list)
    for k, df in dataframe.groupby(["month", "n_site"]):
        cond = np.isfinite(dataframe[clim_col])
        if df[cond].shape[0] < 2:
            continue
        df = df[cond].copy()
        r2 = r2_score(df[y_col], df[y_hat_col])
        rmse = mean_squared_error(df[y_col], df[y_hat_col])
        results_dict["month"].append(k[0])
        results_dict["n_site"].append(k[1])
        results_dict["rmse_pred"].append(np.sqrt(rmse))
        results_dict["r2_score_pred"].append(r2)
        r2_clim = r2_score(df[clim_col], df[y_hat_col])
        rmse_clim = mean_squared_error(df[clim_col], df[y_hat_col])
        results_dict[f"rmse_{clim_col}"].append(np.sqrt(rmse_clim))
        results_dict[f"r2_score_{clim_col}"].append(r2_clim)
        results_dict[f"rmse_skill_{clim_col}"].append(
            1 - np.sqrt(rmse) / np.sqrt(rmse_clim)
        )
        results_dict[f"r2_skill_{clim_col}"].append((r2 - r2_clim) / (1.0 - r2_clim))
    return pd.DataFrame.from_dict(results_dict)
