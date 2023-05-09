from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
import logging
import xarray as xr
import pandas as pd
import numpy as np

from pyproj import Transformer
from sklearn.cluster import DBSCAN


point_xcol = 0
point_ycol = 1

use_dask = True

def load_data(
    f_name,
    verbose=False,
    filter_input_vars=False,
    filter_output_vars=False,
    impute=False,
):

    if verbose:
        logging.info(f"Opening {f_name}")
    if use_dask:
        df = xr.open_dataset(f_name, decode_cf=True, chunks={"n_time":64,"n_site":900})
    else:
        df = xr.open_dataset(f_name, decode_cf=True)
    if verbose:
        logging.info("Converting times to datetime64")
    df["Times"] = df["Times"].astype(str)
    df["Times"].data = pd.to_datetime(df["Times"].data, format="%Y-%m-%d_%H:%M:%S")
    df = df.assign_coords(
        {"n_site": range(0, df.dims["n_site"]), "n_time": df["Times"]}
    )
    if use_dask:
        df = df.to_dask_dataframe().reset_index()
    else:
        df = df.to_dataframe().reset_index()

    if "v4" in f_name:
        f_name = f_name.replace("v4", "v3")
        if verbose:
            logging.info(f"Opening {f_name} b/c v4 is missing band and LST values")
        dg = xr.open_dataset(f_name, decode_cf=True)
        if verbose:
            logging.info("Converting times to datetime64")
        dg["Times"] = dg["Times"].astype(str)
        dg["Times"].data = pd.to_datetime(dg["Times"].data, format="%Y-%m-%d_%H:%M:%S")
        dg = dg.assign_coords(
            {"n_site": range(0, dg.dims["n_site"]), "n_time": dg["Times"]}
        )
        dg = dg.to_dataframe().reset_index()

        new_columns = list(set(df.columns) - set(dg.columns))
        cols = ["n_site", "n_time"] + new_columns
        df = dg.merge(df[cols], on=cols[:2], how="left").copy()

    filter_nans = []
    if isinstance(filter_input_vars, list):
        filter_nans += filter_input_vars
    if isinstance(filter_output_vars, list):
        filter_nans += filter_output_vars
    if len(filter_nans):
        if verbose:
            logging.info(f"Starting df size: {df.shape}")
            logging.info(f"Filtering NaNs from columns: {filter_nans}")
        filter_condition = df[filter_nans].isna().sum(axis=1).astype(bool)
        df = df[~filter_condition].copy()
    if verbose:
        logging.info(f"Total data available for training: {df.shape}")

    if impute:
        if verbose:
            logging.info("Imputing NaNs with the column median")
        df = df.fillna(df.median())

    # if verbose:
    #    logging.info(f"Loaded df NaNs:\n {df[df.isna().any(axis=1)]}")

    return df.reset_index().compute() if use_dask else df.reset_index()


def load_splitter(
    splitter, df, n_splits=1, seed=1000, columns=None, verbose=False, n_jobs=-1
):

    if splitter == "random":
        return data_shuffle_splitter(
            df, n_splits=n_splits, seed=seed, columns=columns, verbose=verbose
        )
    elif splitter == "day":
        return data_splitter(
            df, n_splits=n_splits, seed=seed, columns=["day"], verbose=verbose
        )
    elif splitter == "week":
        return data_splitter(
            df, n_splits=n_splits, seed=seed, columns=["week"], verbose=verbose
        )
    elif splitter == "month":
        return data_splitter(
            df, n_splits=n_splits, seed=seed, columns=["month"], verbose=verbose
        )
    elif splitter == "year":
        return data_splitter(
            df, n_splits=n_splits, seed=seed, columns=["year"], verbose=verbose
        )
    elif splitter == "random-spatial":
        return random_spatial_split(
            df, n_splits=n_splits, seed=seed, columns=columns, verbose=verbose
        )
    elif splitter == "clustered-spatial":
        return clustered_spatial_split(
            df,
            n_splits=n_splits,
            seed=seed,
            columns=columns,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        return None
    else:
        logging.warning(f"Splitting method {splitter} is not supported")
        logging.warning("Choose from: random, day, random-spatial, clustered-spatial")
        raise


def data_splitter(df, n_splits=1, seed=1000, columns=["day"], verbose=False):
    logging.info(f"Splitting the data using the {columns} columns")

    if "day" in columns or "week" in columns or "month" in columns or "year" in columns:
        calendar = df["Times"].dt.isocalendar()
        if "day" in columns:
            df["day"] = calendar["day"].astype(str) + calendar["week"].astype(str) + calendar["year"].astype(str)
        if "week" in columns:
            df["week"] = calendar["week"].astype(str) + calendar["year"].astype(str)
        if "month" in columns:
            df["month"] = df["Times"].dt.month.astype(str) + calendar["year"].astype(str)
        if "year" in columns:
            df["year"] = calendar["year"].astype(str)
        
        #if "day" in columns:
        #    df["day"] = df["Times"].astype(str).apply(lambda x: x.split(" ")[0])

    gsp = GroupShuffleSplit(n_splits=1, random_state=seed, train_size=0.9)
    splits = list(gsp.split(df, groups=df[columns]))
    train_index, test_index = splits[0]
    train_data, test_data = df.iloc[train_index].copy(), df.iloc[test_index].copy()

    # Make N train-valid splits using day as grouping variable
    gsp = GroupShuffleSplit(n_splits=n_splits, random_state=seed, train_size=0.885)
    splits = list(gsp.split(train_data, groups=train_data[columns]))
    for (train_index, valid_index) in splits:
        train_data, valid_data = (
            train_data.iloc[train_index].copy(),
            train_data.iloc[valid_index].copy(),
        )
        yield train_data, valid_data, test_data


def data_shuffle_splitter(df, n_splits=1, seed=1000, columns=None, verbose=False):
    sp = ShuffleSplit(n_splits=1, random_state=seed, train_size=0.9)
    splits = list(sp.split(df))
    train_index, test_index = splits[0]
    train_data, test_data = df.iloc[train_index].copy(), df.iloc[test_index].copy()

    # Make N train-valid splits using day as grouping variable
    sp = ShuffleSplit(n_splits=n_splits, random_state=seed, train_size=0.885)
    splits = list(sp.split(train_data))
    for (train_index, valid_index) in splits:
        train_data, valid_data = (
            df.iloc[train_index].copy(),
            df.iloc[valid_index].copy(),
        )
        yield train_data, valid_data, test_data


def random_spatial_split(df, n_splits=1, seed=1000, columns=None, verbose=False):
    """
    Performs a random sampling of unique lat,lon site coords to create training, validation, and test set DataFrames
    """

    if verbose:
        logging.info("Performing spatial split")
    df["labels"] = df.groupby(["latitude", "longitude"]).ngroup()

    uniq_labels = set(df["labels"])
    if verbose:
        logging.info(f"Number of unique labels: {len(uniq_labels)}")

    sp = GroupShuffleSplit(n_splits=1, random_state=seed, train_size=0.9)
    splits = list(sp.split(df, groups=df["labels"]))
    train_index, test_index = splits[0]
    train_data, test_data = df.iloc[train_index].copy(), df.iloc[test_index].copy()

    # Make N train-valid splits using day as grouping variable
    sp = GroupShuffleSplit(n_splits=n_splits, random_state=seed, train_size=0.885)
    splits = list(sp.split(train_data, groups=train_data["labels"]))
    for (train_index, valid_index) in splits:
        train_data, valid_data = (
            df.iloc[train_index].copy(),
            df.iloc[valid_index].copy(),
        )
        yield train_data, valid_data, test_data


def wgs84_2_lcc(coords, loncol=point_xcol, latcol=point_ycol):
    """
    Converts WGS84 geographic coordinates to Lambert Conformal Conic
    """
    # epsg:4326 = WGS84
    lcc_epsg = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs"
    transformer = Transformer.from_crs("epsg:4326", lcc_epsg, always_xy=True)
    fx, fy = transformer.transform(coords[:, loncol], coords[:, latcol])
    return fx, fy


def clustered_spatial_split(
    df, n_splits=1, seed=1000, columns=None, verbose=False, n_jobs=-1
):
    """
    Peforms a random sampling of clusters of nearby sites to create training, validation, and test set DataFrames
    """

    if verbose:
        print("Performing spatial split")
    coords = np.array(list(zip(df.longitude, df.latitude)))

    xs, ys = wgs84_2_lcc(coords)
    xyz_coords = np.array(list(zip(xs, ys, df.HGT_M)))
    coords_df = pd.DataFrame(xyz_coords, columns=["x", "y", "z"])

    uniq_xyz_coords = np.array(list(set(tuple(zip(xs, ys, df.HGT_M)))))
    label_coords_df = pd.DataFrame(uniq_xyz_coords, columns=["x", "y", "z"])

    if verbose:
        print("Clustering sites...")
    db = DBSCAN(eps=16000, min_samples=2, n_jobs=n_jobs).fit(uniq_xyz_coords)

    label_coords_df["labels"] = db.labels_
    assert len(label_coords_df[label_coords_df.isna().any(axis=1)]) == 0

    coords_df = coords_df.merge(label_coords_df, on=["x", "y", "z"])
    assert len(coords_df[coords_df.isna().any(axis=1)]) == 0

    df["labels"] = coords_df["labels"]
    assert len(df[df["labels"].isna()]) == 0

    uniq_labels = set(df["labels"])
    if verbose:
        logging.info(f"Unique labels: {uniq_labels}")

    sp = GroupShuffleSplit(n_splits=1, random_state=seed, train_size=0.9)
    splits = list(sp.split(df, groups=df["labels"]))
    train_index, test_index = splits[0]
    train_data, test_data = df.iloc[train_index].copy(), df.iloc[test_index].copy()

    # Make N train-valid splits using day as grouping variable
    sp = GroupShuffleSplit(n_splits=n_splits, random_state=seed, train_size=0.885)
    splits = list(sp.split(train_data, groups=train_data["labels"]))
    for (train_index, valid_index) in splits:
        train_data, valid_data = (
            train_data.iloc[train_index].copy(),
            train_data.iloc[valid_index].copy(),
        )
        yield train_data, valid_data, test_data


def vegetatation_indices(df):
    # Normalized difference vegetation index
    df["ndvi"] = (
        df["375m Surface Reflectance Band I2_medium"]
        - df["375m Surface Reflectance Band I1_medium"]
    )
    df["ndvi"] /= (
        df["375m Surface Reflectance Band I2_medium"]
        + df["375m Surface Reflectance Band I1_medium"]
    )
    # Normalized difference water index
    df["ndwi_1"] = (
        df["375m Surface Reflectance Band I2_medium"]
        - df["750m Surface Reflectance Band M8_medium"]
    )
    df["ndwi_1"] /= (
        df["375m Surface Reflectance Band I2_medium"]
        + df["750m Surface Reflectance Band M8_medium"]
    )
    df["ndwi_2"] = (
        df["750m Surface Reflectance Band M7_medium"]
        - df["750m Surface Reflectance Band M8_medium"]
    )
    df["ndwi_2"] /= (
        df["750m Surface Reflectance Band M7_medium"]
        + df["750m Surface Reflectance Band M8_medium"]
    )
    # Normalized difference Infrared index
    # df["ndii_1"] = (df["750m Surface Reflectance Band M6_medium"] - df["750m Surface Reflectance Band M10_medium"])
    # df["ndii_1"] /= (df["750m Surface Reflectance Band M6_medium"] + df["750m Surface Reflectance Band M10_medium"])
    df["ndii_2"] = (
        df["750m Surface Reflectance Band M7_medium"]
        - df["750m Surface Reflectance Band M10_medium"]
    )
    df["ndii_2"] /= (
        df["750m Surface Reflectance Band M7_medium"]
        + df["750m Surface Reflectance Band M10_medium"]
    )
    # Enhanced Vegetation Index
    df["evi"] = 2.5 * (
        df["375m Surface Reflectance Band I2_medium"]
        - df["375m Surface Reflectance Band I1_medium"]
    )
    df["evi"] /= (
        df["375m Surface Reflectance Band I2_medium"]
        + 6 * df["375m Surface Reflectance Band I1_medium"]
        - 7.5 * df["750m Surface Reflectance Band M3_medium"]
        + 1
    )
    # Normalized difference tillage index
    df["ndti"] = (
        df["750m Surface Reflectance Band M10_medium"]
        - df["750m Surface Reflectance Band M11_medium"]
    )
    df["ndti"] /= (
        df["750m Surface Reflectance Band M10_medium"]
        + df["750m Surface Reflectance Band M11_medium"]
    )
    # Visible Atmospherically Resistant Index
    df["vari_1"] = (
        df["750m Surface Reflectance Band M4_medium"]
        - df["375m Surface Reflectance Band I1_medium"]
    )
    df["vari_1"] /= (
        df["750m Surface Reflectance Band M4_medium"]
        + df["375m Surface Reflectance Band I1_medium"]
        - df["750m Surface Reflectance Band M3_medium"]
    )
    df["vari_2"] = (
        df["750m Surface Reflectance Band M4_medium"]
        - df["750m Surface Reflectance Band M5_medium"]
    )
    df["vari_2"] /= (
        df["750m Surface Reflectance Band M4_medium"]
        + df["750m Surface Reflectance Band M5_medium"]
        - df["750m Surface Reflectance Band M3_medium"]
    )
    # Global Environmental Monitoring Index
    I1 = df["375m Surface Reflectance Band I1_medium"]
    I2 = df["375m Surface Reflectance Band I2_medium"]
    eta = 2 * (2 * (I2**2 - I1**2) + 1.5 * I2 + 0.5 * I1) / (I2 + I1 + 0.5)
    df["gemi"] = eta * (1 - 0.25 * eta) - (I1 - 0.125) / (1 - I1)
    # Moisture Stress Index
    df["msi_1"] = (
        df["375m Surface Reflectance Band I3_medium"]
        / df["375m Surface Reflectance Band I2_medium"]
    )
    df["msi_2"] = (
        df["375m Surface Reflectance Band I3_medium"]
        / df["750m Surface Reflectance Band M7_medium"]
    )
    # Normalized multiband drought index
    I3 = df["375m Surface Reflectance Band I3_medium"]
    M11 = df["750m Surface Reflectance Band M7_medium"]
    df["nmdi"] = (I2 - (I3 - M11)) / (I2 + (I3 - M11))

    veg_group = [
        "ndvi",
        "ndwi_1",
        "ndwi_2",
        "ndii_2",
        "evi",
        "ndti",
        "vari_1",
        "vari_2",
        "gemi",
        "msi_1",
        "msi_2",
        "nmdi",
    ]

    return df, veg_group
