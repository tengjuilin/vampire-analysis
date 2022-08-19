import os
import re

import cv2
import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import regionprops_table

from . import util


def check_property_csv_existence(img_set_path, filter_info):
    """
    Check existence of property csv that contain object properties.

    Parameters
    ----------
    img_set_path : str
        Path to the directory of images to be analyzed.
    filter_info : ndarray
        Regex filter(s) of image filenames to be analyzed.
        Empty if no filter needed.

    Returns
    -------
    bool

    """
    properties_csv_path = util.get_properties_csv_path(img_set_path, filter_info)
    if os.path.exists(properties_csv_path):
        print(f'Contour and properties data already exist in path: {img_set_path}')
        return True
    return False


def get_filtered_filenames(img_set_path, filter_info=None):
    """
    Get filenames filtered with keywords.

    Parameters
    ----------
    img_set_path : str
        Path to the directory of images to be analyzed.
    filter_info : ndarray, optional
        Regex filter(s) of image filenames to be analyzed.
        Empty if no filter needed.

    Returns
    -------
    filtered_filenames : ndarray
        Filtered filenames.

    """
    if filter_info is None:
        filter_info = np.array([], dtype=str)

    filenames = pd.Series(os.listdir(img_set_path))

    # filter by img extension
    extensions_regex = r'\.tif|\.jpeg|\.jpg|\.png|\.bmp|\.gif'
    extension_filter = filenames.str.contains(extensions_regex,
                                              flags=re.IGNORECASE)
    filenames = filenames[extension_filter]

    # filter by user constraints
    for constraint in filter_info:
        constraint_filter = filenames.str.contains(constraint, regex=True)
        filenames = filenames[constraint_filter]
    filenames = np.array(filenames)
    return filenames


def get_img_set(img_set_path, filenames):
    """
    Get an image set from image set path.

    Parameters
    ----------
    img_set_path : str
        Path to the directory of images to be analyzed.
    filenames : ndarray
        Filtered filenames.

    Returns
    -------
    img_set: list[ndarray]
        A list of images to be analyzed.

    """
    img_set = []
    for filename in filenames:
        # read image and get contours and properties
        img = io.imread(os.path.join(img_set_path, filename))
        img_set.append(img)
    return img_set


def extract_contour_from_object(object_img):
    """
    Returns x and y coordinates of the object contour.

    Parameters
    ----------
    object_img : ndarray
        2D binary image with only one object.

    Returns
    -------
    contour : ndarray
        x and y coordinates of n contour sample points, with shape (2, n)

    """
    contour = cv2.findContours(
        object_img.astype('uint8'),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )[0][0]
    contour = contour.reshape(-1, 2).T
    contour = np.flip(contour, axis=1)
    if contour.size <= 6:  # contour has <= 3 points, could not be sampled
        return np.nan
    return contour


def extract_properties_from_img(img, filename=None, img_id=None):
    """
    Get object properties of all objects in an image.

    Parameters
    ----------
    img : ndarray
        Image to be analyzed
    filename : str, optional
        Filename of the image.
    img_id : int, optional
        ID/index of the image.

    Returns
    -------
    properties_df : DataFrame
        Dataframe of object properties.

    """
    # get properties of objects
    properties = (
        'label',
        'centroid',
        'area',
        'bbox_area',
        'convex_area',
        'filled_area',
        'perimeter',
        'equivalent_diameter',
        'major_axis_length',
        'minor_axis_length',
        'orientation',
        'euler_number',
        'eccentricity',
        'solidity',
        'extent'
    )
    properties_dict = regionprops_table(
        img,
        properties=properties,
        extra_properties=(extract_contour_from_object,)
    )
    properties_df = pd.DataFrame(properties_dict)
    properties_df.rename(
        columns={
            'centroid-0': 'centroid-y',
            'centroid-1': 'centroid-x',
            'extract_contour_from_object': 'raw_contour'
        },
        inplace=True
    )
    # additional properties
    properties_df['circularity'] = 4 * np.pi * properties_df['area'] / properties_df['perimeter'] ** 2
    properties_df['aspect_ratio'] = np.nan_to_num(np.divide(
        properties_df['major_axis_length'],
        properties_df['minor_axis_length']
    ))
    # discard contours with <= 3 points that cannot be sampled
    properties_df = properties_df[pd.notna(properties_df['raw_contour'])]
    # label each object
    if img_id is not None:
        properties_df.insert(0, 'image_id', img_id)
    if filename is not None:
        properties_df.insert(0, 'filename', filename)
    return properties_df


def extract_properties_from_img_set(img_set, filenames=None):
    """
    Get object properties of all objects in an image set.

    Parameters
    ----------
    img_set: list[ndarray]
        A list of images to be analyzed.
    filenames : ndarray, optional
        Filenames of the images.

    Returns
    -------
    properties_from_img_set_df : DataFrame
        Dataframe of object properties.

    """
    if filenames is not None and len(img_set) != len(filenames):
        raise ValueError('Length of img_set and filenames does not match.')
    properties_from_img_set = []
    for img_i, img in enumerate(img_set):
        if filenames is not None:
            filename = filenames[img_i]
        else:
            filename = None
        properties_from_img = extract_properties_from_img(
            img,
            filename=filename,
            img_id=img_i
        )
        properties_from_img_set.append(properties_from_img)
    properties_from_img_set_df = pd.concat(
        properties_from_img_set,
        ignore_index=True
    )
    return properties_from_img_set_df


def read_properties(img_set_path, filter_info):
    """
    Read object properties from existing property ``pickle`` file.

    Parameters
    ----------
    img_set_path : str
        Path to the directory of images to be analyzed.
    filter_info : ndarray
        Regex filter(s) of image filenames to be analyzed.
        Empty if no filter needed.

    Returns
    -------
    properties_df : DataFrame
        Dataframe of object properties.

    """
    properties_pickle_path = util.get_properties_pickle_path(img_set_path, filter_info)
    properties_df = util.read_pickle(properties_pickle_path)
    return properties_df


def write_properties(properties_df, img_set_path, filter_info):
    """
    Writes contour coordinates and properties to given paths.

    Parameters
    ----------
    properties_df : DataFrame
        DataFrame of object properties.
    img_set_path : str
        Path to the directory of images to be analyzed.
    filter_info : ndarray
        Regex filter(s) of image filenames to be analyzed.
        Empty if no filter needed.

    """
    properties_csv_path = util.get_properties_csv_path(img_set_path, filter_info)
    properties_pickle_path = util.get_properties_pickle_path(img_set_path, filter_info)
    properties_df.drop('raw_contour', axis=1).to_csv(properties_csv_path, index=False)
    util.write_pickle(properties_pickle_path, properties_df)
    return


def extract_properties(img_set_path, filter_info=None, write=True):
    """
    Extracts object properties from image set path.

    Parameters
    ----------
    img_set_path : str
        Path to the directory of images to be analyzed.
    filter_info : ndarray, optional
        Regex filter(s) of image filenames to be analyzed.
        Empty if no filter needed.
    write : bool
        Write properties into ``csv`` and ``pickle`` file.

    Returns
    -------
    properties_df : Dataframe
        Dataframe of object properties.

    """
    empty_filter = np.array([], dtype=str)
    if filter_info is None:
        filter_info = empty_filter
    full_set_exist = check_property_csv_existence(img_set_path, empty_filter)
    specific_set_exist = check_property_csv_existence(img_set_path, filter_info)

    if specific_set_exist:
        properties_df = read_properties(img_set_path, filter_info)
    elif full_set_exist:
        # extract specific set info from full set
        filenames = get_filtered_filenames(img_set_path, filter_info)
        full_properties_df = read_properties(img_set_path, empty_filter)
        filename_filter = np.isin(full_properties_df['filename'], filenames)
        properties_df = full_properties_df[filename_filter].reset_index(drop=True)
        if write:
            write_properties(properties_df, img_set_path, filter_info)
    else:
        filenames = get_filtered_filenames(img_set_path, filter_info)
        img_set = get_img_set(img_set_path, filenames)
        properties_df = extract_properties_from_img_set(
            img_set,
            filenames=filenames
        )
        if write:
            write_properties(properties_df, img_set_path, filter_info)
    return properties_df
