import os
import pickle

import numpy as np
import pandas as pd


def read_pickle(path):
    """
    Loads content in the pickle file from `path`.

    Parameters
    ----------
    path : str
        Path of pickle file to be loaded.

    Returns
    -------
    content
        Content of the pickle file.

    """
    path = os.path.normpath(path)
    opened_file = open(path, 'rb')
    content = pickle.load(opened_file)
    opened_file.close()
    return content


def write_pickle(path, variable):
    """
    Writes `variable` to `path` as a pickle file.

    Parameters
    ----------
    path : str
        Path of pickle file to be saved.
    variable
        A variable to be saved in pickle file.

    """
    path = os.path.normpath(path)
    opened_file = open(path, 'wb')
    pickle.dump(variable, opened_file)
    opened_file.close()


def generate_file_paths(filepath, filename, filter_info, extension):
    """
    Generates file paths according to regex filters.

    Parameters
    ----------
    filepath : str
        Path to the file.
    filename : str
        Common filename of the file.
    filter_info : ndarray
        Regex filter(s) of image filenames to be analyzed.
        Empty if no filter needed.
    extension : str
        Extension of the file, including the dot `.`.

    Returns
    -------
    file_path : str
        Path to named file.

    """
    filter_info = pd.Series(filter_info)
    prohibited_char_regex = r'(\\)|(\/)|(\,)|(\:)|(\*)|(\")|(\<)|(\>)|(\|)'
    replacement = '-'
    filter_info = filter_info.str.replace(prohibited_char_regex, replacement, regex=True)
    filter_tag = '_'.join(filter_info)

    file_path = os.path.join(filepath, f'{filename}__{filter_tag}{extension}')
    return file_path


def get_properties_pickle_path(filepath, filter_info):
    return generate_file_paths(filepath, 'raw-properties', filter_info, '.pickle')


def get_properties_csv_path(filepath, filter_info):
    return generate_file_paths(filepath, 'raw-properties', filter_info, '.csv')


def get_model_pickle_path(filepath, filter_info, model):
    model_name = model.model_name
    n_points = model.n_points
    n_clusters = model.n_clusters
    n_pcs = model.n_pcs
    return generate_file_paths(
        filepath,
        f'model_{model_name}_({n_points}_{n_clusters}_{n_pcs})',
        filter_info,
        '.pickle'
    )


def get_apply_properties_csv_path(filepath, filter_info, model, img_set_name):
    model_name = model.model_name
    n_points = model.n_points
    n_clusters = model.n_clusters
    n_pcs = model.n_pcs
    return generate_file_paths(
        filepath,
        f'apply-properties_{model_name}_on_{img_set_name}_({n_points}_{n_clusters}_{n_pcs})',
        filter_info,
        '.csv'
    )


def get_apply_properties_pickle_path(filepath, filter_info, model, img_set_name):
    model_name = model.model_name
    n_points = model.n_points
    n_clusters = model.n_clusters
    n_pcs = model.n_pcs
    return generate_file_paths(
        filepath,
        f'apply-properties_{model_name}_on_{img_set_name}_({n_points}_{n_clusters}_{n_pcs})',
        filter_info,
        '.pickle'
    )


def label_df(properties_df, id_df, target_props, search_prop="filename"):
    """

    Parameters
    ----------
    properties_df : DataFrame
        Contains "filename" column
    id_df : DataFrame
        Contains columns listed in `properties` and "regex" column.
    target_props : list
        List of property(s) to be added. Must match column name in `id_df`.
    search_prop : str, optional


    Returns
    -------

    """
    properties_df = properties_df.copy()
    properties_df[target_props] = np.nan
    for i in range(id_df["regex"].size):
        mask = properties_df[search_prop].astype(str).str.contains(id_df["regex"][i])
        properties_df.loc[mask, target_props] = id_df.iloc[i, :][target_props].values
    return properties_df
