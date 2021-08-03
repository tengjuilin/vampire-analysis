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
    opened_file = open(path, 'wb')
    pickle.dump(variable, opened_file)
    opened_file.close()


def generate_file_paths(img_set_path, filter_info):
    """
    Returns paths to contour ``pickle`` file and property ``csv`` file.

    Parameters
    ----------
    img_set_path : str
        Path to the directory of images to be analyzed.
    filter_info : ndarray
        Unique filter(s) of image filenames to be analyzed. Empty if no filter
        needed.

    Returns
    -------
    contours_pickle_path : str
        Path to contour ``pickle`` file.
    properties_csv_path : str
        Path to property ``csv`` file.

    """
    filter_tag = '_'.join(filter_info)
    contours_pickle_path = os.path.join(img_set_path, f'contour_coordinates__{filter_tag}.pickle')
    properties_csv_path = os.path.join(img_set_path, f'vampire_datasheet__{filter_tag}.csv')
    return contours_pickle_path, properties_csv_path


def write_clusters_info(img_set_path, filter_info, contours_df, distance):
    """
    Writes cluster id (closest centroid) and distance to closest centroid
    to property csv file.

    Parameters
    ----------
    img_set_path : str
        Path of folder that contains images to be analyzed.
    filter_info : ndarray
        Unique filter(s) of image filenames to be analyzed.
        Empty if no filter is needed.
    contours_df : DataFrame
        DataFrame of objects' contour coordinates with cluster id.
    distance : ndarray
        Distance of truncated principal components to the closest centroid.

    """
    _, properties_csv_path = generate_file_paths(img_set_path, filter_info)
    properties_df = pd.read_csv(properties_csv_path)
    properties_df['cluster_id'] = contours_df['cluster_id']
    properties_df['distance_to_centroid'] = distance
    properties_df.to_csv(properties_csv_path, index=False)
    return
