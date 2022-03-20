import os
import pickle

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
    num_points = model.num_points
    num_clusters = model.num_clusters
    num_pc = model.num_pc
    return generate_file_paths(filepath, f'model_{model_name}_({num_points}_{num_clusters}_{num_pc})',
                               filter_info,
                               '.pickle')


def get_apply_properties_csv_path(filepath, filter_info, model, img_set_name):
    model_name = model.model_name
    num_points = model.num_points
    num_clusters = model.num_clusters
    num_pc = model.num_pc
    return generate_file_paths(filepath, f'apply-properties_{model_name}_on_{img_set_name}_({num_points}_{num_clusters}_{num_pc})',
                               filter_info,
                               '.csv')


def get_apply_properties_pickle_path(filepath, filter_info, model, img_set_name):
    model_name = model.model_name
    num_points = model.num_points
    num_clusters = model.num_clusters
    num_pc = model.num_pc
    return generate_file_paths(filepath, f'apply-properties_{model_name}_on_{img_set_name}_({num_points}_{num_clusters}_{num_pc})',
                               filter_info,
                               '.pickle')
