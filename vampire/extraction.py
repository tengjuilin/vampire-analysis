import os

import cv2
import numpy as np
import pandas as pd
from skimage import io
from skimage.measure import regionprops_table

from . import util


def _is_filtered_img(filename, filter_info):
    """
    Checks if the file is a tagged image.

    Supports image extensions ``'.tiff', '.tif', '.jpeg', '.jpg', '.png',
    '.bmp', '.gif'``.

    Parameters
    ----------
    filename : str
        Filename of file to be checked.
    filter_info : ndarray
        Unique filter(s) of image filenames to be analyzed.
        Empty if no filter is needed.

    Returns
    -------
    bool

    """
    extensions = ('.tiff', '.tif', '.jpeg', '.jpg', '.png', '.bmp', '.gif')
    for filter_name in filter_info:
        if not (filter_name in filename):
            return False
    if not filename.endswith(extensions):
        return False
    return True


def _generate_file_paths(img_set_path, filter_info):
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
    contours_pickle_path = os.path.join(img_set_path, f'boundary_coordinates__{filter_tag}.pickle')
    properties_csv_path = os.path.join(img_set_path, f'vampire_datasheet__{filter_tag}.csv')
    return contours_pickle_path, properties_csv_path


def get_contour_from_object(object_img):
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
    contour = cv2.findContours(object_img.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    contour = contour.reshape(-1, 2).T
    # contour = contour + 1  # original code does this, but I think there's no reason to do that
    contour = np.flip(contour, axis=1)

    # my alternative option
    # from skimage.measure import find_contours
    # contour = find_contours(object_img, fully_connected='high')[0].T
    # contour = np.flip(contour, axis=0)  # switch convention from (row, col) to (x, y)

    # testing visualization
    # plt.plot(*contour, '.-', alpha=0.5)
    # plt.axis('equal')
    return contour


def get_contours_from_image(img, object_labels):
    """
    Returns a list of contours associated with each object in an image.

    Parameters
    ----------
    img : ndarray
        2D binary image to be analyzed.
    object_labels : ndarray
        Labels of each object.

    Returns
    -------
    contours : list[ndarray]
        List of contour coordinates of objects.

    """
    contours = []
    for object_label in object_labels:
        object_img = (img == object_label)
        contour = get_contour_from_object(object_img)
        if contour.size < 10:  # 5 points offer poor resolution, throw away
            continue
        contours.append(contour)

    # testing visualization
    # for contour in contours:
    #     plt.plot(*contour, '.-', alpha=0.5)
    # plt.imshow(img != 0)
    # plt.axis('equal')
    return contours


def get_properties_from_image(img, filename, image_index, object_labels):
    """
    Returns properties of the objects in the images.

    Parameters
    ----------
    img : ndarray
        2D grayscale labeled image of objects.
    filename : str
        Filename of the image where the objects come from.
    image_index : int
        ID of the image.
    object_labels : ndarray
        Labels of the objects.

    Returns
    -------
    properties_df : DataFrame
        Properties of each object in the image with labels.

    """
    # get properties of objects
    properties = ('label', 'centroid', 'area', 'perimeter', 'major_axis_length', 'minor_axis_length')
    properties_df = pd.DataFrame(regionprops_table(img, properties=properties)).set_index('label')
    properties_df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'}, inplace=True)
    properties_df['circularity'] = 4 * np.pi * properties_df['area'] * properties_df['perimeter'] ** 2
    properties_df['aspect_ratio'] = np.nan_to_num(np.divide(properties_df['major_axis_length'],
                                                            properties_df['minor_axis_length']))
    # label each object
    properties_df.insert(0, 'filename', filename)
    properties_df.insert(1, 'image_id', image_index)
    object_labels = pd.Series(np.arange(len(object_labels)) + 1, index=object_labels)
    properties_df.insert(2, 'object_id', object_labels)
    return properties_df


def get_info_from_folder(img_set_path, filter_info):
    """
    Returns contour and properties of objects from the image set folder.

    Parameters
    ----------
    img_set_path : str
        Path of folder that contains images to be analyzed.
    filter_info : ndarray
        Unique filter(s) of image filenames to be analyzed.
        Empty if no filter is needed.

    Returns
    -------
    contours_from_folder : list[ndarray]
        List of ndarray of contour coordinates of objects
    properties_from_folder : list[DataFrame]
        List of DataFrames of properties of objects

    """
    contours_from_folder = []
    properties_from_folder = []
    img_i = 1
    filenames = np.char.lower(np.array(os.listdir(img_set_path)))
    for filename in filenames:
        # only images containing filter info proceed to calculations below
        filtered_img = _is_filtered_img(filename, filter_info)
        if not filtered_img:
            continue
        # read image and get contours and properties
        img = io.imread(os.path.join(img_set_path, filename))
        object_labels = np.unique(img)[1:]
        contours_from_img = get_contours_from_image(img, object_labels)
        properties_from_img = get_properties_from_image(img, filename, img_i, object_labels)
        contours_from_folder.extend(contours_from_img)
        properties_from_folder.append(properties_from_img)
        img_i += 1
    return contours_from_folder, properties_from_folder


def write_contours(img_set_path, filter_info):
    """
    Finds contour coordinates of objects in the images from the image set.

    Parameters
    ----------
    img_set_path : str
        Path to the directory of images to be analyzed.
    filter_info : ndarray
        Unique filter(s) of image filenames to be analyzed. Empty if no filter
        needed.

    Returns
    -------
    contours : list[ndarray]
        List of ndarray of contour coordinates.

    """
    # calculations
    contours, properties = get_info_from_folder(img_set_path, filter_info)
    # write the files
    contours_pickle_path, properties_csv_path = _generate_file_paths(img_set_path, filter_info)
    util.write_pickle(contours_pickle_path, contours)
    pd.concat(properties).to_csv(properties_csv_path)
    return contours


def read_contours(contours_pickle_path):
    """
    Retrieves contour coordinates from existing contour ``pickle`` file.

    Parameters
    ----------
    contours_pickle_path : str
        Path to contour ``pickle`` file.

    Returns
    -------
    contours : list[ndarray]
        List of ndarray of contour coordinates.

    """
    contours = util.read_pickle(contours_pickle_path)
    return contours


# main
def extract_contours(img_set_path, filter_info):
    """
    Returns contour coordinates of objects in the images from the image set.

    Parameters
    ----------
    img_set_path : str
        Path to the directory of images to be analyzed.
    filter_info : ndarray
        Unique filter(s) of image filenames to be analyzed. Empty if no filter
        needed.

    Returns
    -------
    contours : list[ndarray]
        List of ndarray of contour coordinates.

    """
    contours_pickle_path, properties_csv_path = _generate_file_paths(img_set_path, filter_info)
    if os.path.exists(contours_pickle_path) and os.path.exists(properties_csv_path):
        print(f'Contour and properties data already exist in path: {img_set_path}')
        contours = read_contours(contours_pickle_path)
        return contours
    else:
        contours = write_contours(img_set_path, filter_info)
        return contours


# def collect_contours(csv_path):
#     """
#     Collect contour coordinates from pickle.
#
#     Originally `collect_seleced_bstack()`.
#
#     Parameters
#     ----------
#     csv_path : str
#         Path of the `csv` file that stores information about image set
#         used to build model.
#
#     Returns
#     -------
#     contours : list
#
#     """
#     img_df = pd.read_csv(csv_path)
#     folder_paths = img_df['set location']
#     tags = img_df['tag']
#     contours = []
#     for path_i, folder_path in enumerate(folder_paths):
#         pickles = [pickle_file
#                    for pickle_file in os.listdir(folder_path)
#                    if is_tagged_pickle(pickle_file, tags[path_i])]
#         contours = []
#         for pickle_i in pickles:
#             if tags[path_i] in pickle_i:
#                 contours.append(read_pickle(os.path.join(folder_path, pickle_i)))
#     return contours


# def generate_contour_properties_files(img_set_path, filter_info):
#     """
#     Generates a pickle file containing a list of contour coordinates of objects
#     in the image set.
#
#     Originally `getboundary()`
#
#     Parameters
#     ----------
#     img_set_path : str
#         Path to the directory of images to be analyzed.
#     filter_info : ndarray
#         Unique filter(s) of image filenames to be analyzed. Empty if no filter
#         needed.
#
#     """
#     filter_tag = '_'.join(filter_info)
#     contours_pickle_path = os.path.join(img_set_path, f'boundary_coordinates__{filter_tag}.pickle')
#     properties_csv_path = os.path.join(img_set_path, f'vampire_datasheet__{filter_tag}.csv')
#
#     # check existence of previous calculation
#     if os.path.exists(contours_pickle_path) and os.path.exists(properties_csv_path):
#         print(f'Contour and properties data already exist in path: {img_set_path}')
#         return
#
#     # calculations
#     contours, properties = get_contours_and_properties_from_folder(img_set_path, filter_info)
#     # generates the files
#     util.write_pickle(contours_pickle_path, contours)
#     pd.concat(properties).to_csv(properties_csv_path)
#     return
