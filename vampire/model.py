import os
from datetime import datetime

import numpy as np
import pandas as pd

from . import analysis
from . import extraction
from . import plot
from . import processing
from . import util


def _check_prohibited_char(text, input_type='path'):
    r"""
    Checks if path contains characters prohibited by the operating system.

    Parameters
    ----------
    text : str
        Path of file or directory.
    input_type : str, optional
        Type of input text. Default ``'path'``.

            ``'path'``
                Input text as a path (of either directory or filename).
                Should not contain any of ``,*"<>|``

            ``'file'``
                Input text as filename only.
                Should not contain any of ``\/,:*"<>|``

    Raises
    ------
    ValueError
        If text contains prohibited character.

    """
    if input_type == 'path':
        prohibited_chars = [',', '*', '"', '<', '>', '|']
    elif input_type == 'file':
        prohibited_chars = ['\\', '/', ',', ':', '*', '"', '<', '>', '|']
    else:
        raise ValueError('Unrecognized input_type: {input_type}'
                         'Expect one in {"path", "file"}')
    have_prohibited_char = any(prohibited_char in text
                               for prohibited_char in prohibited_chars)
    if have_prohibited_char:
        raise ValueError(f'Filename-related entry of {text} contains prohibited character(s). \n'
                         'Expect a model name without any of the following characters: \n'
                         '\\  /  ,  :  *  "  <  >  |')


def _build_models_parse_df(img_info_df):
    """
    Checks if input DataFrame to `build_models` has the appropriate shape.

    Parameters
    ----------
    img_info_df : DataFrame
        Input to `build_models`. Contains all information about image sets
        to be analyzed.

    Raises
    ------
    ValueError
        Empty DataFrame without information in rows.
    ValueError
        DataFrame does not contain required columns specified in the doc.

    """
    num_img_sets, num_args = img_info_df.shape
    if num_img_sets == 0:
        raise ValueError('Input DataFrame is empty. Expect at least one row.')
    if num_args < 5:  # 5 cols required by doc
        raise ValueError('Input DataFrame does not have enough number of columns. \n'
                         'Expect required 5 columns in order: img_set_path, output_path, '
                         'model_name, num_points, num_clusters.')


def _build_models_parse_required_info(required_info):
    """
    Parse required columns of input DataFrame to `build_models`.

    Checks argument requirements and sets default arguments.

    Parameters
    ----------
    required_info : DataFrame
        Required columns (1-5) of input DataFrame ``img_info_df``.

    Returns
    -------
    img_set_path : str
        Path to the directory containing the image set(s) used to build model.
    output_path : str
        Path of the directory used to output model and figures. Defaults to
        ``img_set_path``.
    model_name : str
        Name of the model. Defaults to time of function call.
    num_points : int
        Number of sample points of object contour. Defaults to 50.
    num_clusters : int
        Number of clusters of K-means clustering. Defaults to 5.

    Raises
    ------
    FileNotFoundError
        If ``img_set_path`` does not exist.

    """
    # unpack args
    img_set_path, output_path, model_name, num_points, num_clusters = required_info

    # img_set_path
    if os.path.isdir(img_set_path):
        img_set_path = os.path.normpath(img_set_path)
    else:
        raise FileNotFoundError(f'Input DataFrame column 1 gives non-existing directory: \n'
                                f'{img_set_path} \n'
                                'Expect an existing directory with images used to build model.')

    # output_path
    if pd.isna(output_path):
        output_path = os.path.normpath(img_set_path)  # default
    else:
        _check_prohibited_char(output_path)
        output_path = os.path.normpath(output_path)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

    # model_name
    if pd.isna(model_name):
        time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_name = time_stamp
    else:
        _check_prohibited_char(model_name, 'file')

    # num_points
    if pd.isna(num_points):
        num_points = 50
    else:
        num_points = int(num_points)

    # num_clusters
    if pd.isna(num_clusters):
        num_clusters = 5
    else:
        num_clusters = int(num_clusters)

    return img_set_path, output_path, model_name, num_points, num_clusters


def _apply_models_parse_df(img_info_df):
    """
    Checks if input DataFrame to `apply_models` has the appropriate shape.

    Parameters
    ----------
    img_info_df : DataFrame
        Input to `apply_models`. Contains all information about image sets
        to be analyzed.

    Raises
    ------
    ValueError
        Empty DataFrame without information in rows.
    ValueError
        DataFrame does not contain required columns specified in the doc.

    """
    num_img_set, num_args = img_info_df.shape
    if num_img_set == 0:
        raise ValueError('Input DataFrame is empty. Expect at least one row.')
    if num_args < 4:  # 4 cols required by doc
        raise ValueError('Input DataFrame does not have enough number of columns. \n'
                         'Expect required 3 columns in order: img_set_path, model_path, '
                         'output_path.')
    return


def _apply_models_parse_required_info(required_info):
    """
    Parse required columns of input DataFrame to `apply_models`.

    Parameters
    ----------
    required_info : DataFrame
        Required columns (1-4) of input DataFrame ``img_info_df``.

    Returns
    -------
    img_set_path : str
        Path to the directory containing the image set(s) used to apply model.
    model_path : str
        Path to the pickle file that stores model information.
    output_path : str
        Path of the directory used to output model and figures. Defaults to
        ``img_set_path``.
    img_set_name : str
        Name of the image set being applied to.
        Defaults to time of function call.

    Raises
    ------
    FileNotFoundError
        If ``img_set_path`` does not exist.
    FileNotFoundError
        If ``model_path`` does not exist.
    ValueError
        If ``model_path`` is not a ``pickle`` file.

    """
    # unpack args
    img_set_path, model_path, output_path, img_set_name = required_info

    # img_set_path
    if os.path.isdir(img_set_path):
        img_set_path = os.path.normpath(img_set_path)
    else:
        raise FileNotFoundError(f'Input DataFrame column 1 gives non-existing directory: \n'
                                f'{img_set_path} \n'
                                'Expect an existing directory with images used to apply model.')

    # model_path
    if os.path.isfile(model_path):
        filename, extension = os.path.splitext(model_path)
        if extension == '.pickle':
            model_path = os.path.normpath(model_path)
        else:
            raise ValueError(f'Input DataFrame column 2 gives non-pickle file: \n'
                             f'{model_path} \n'
                             'Expect an existing pickle file for model information.')
    else:
        raise FileNotFoundError(f'Input DataFrame column 2 gives non-existing file: \n'
                                f'{model_path} \n'
                                'Expect an existing pickle file for model information.')

    # output_path
    if pd.isna(output_path):
        output_path = os.path.normpath(img_set_path)
    else:
        _check_prohibited_char(output_path)
        output_path = os.path.normpath(output_path)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

    # img_set_name
    if pd.isna(img_set_name):
        time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        img_set_name = time_stamp
    else:
        _check_prohibited_char(img_set_name, 'file')

    return img_set_path, model_path, output_path, img_set_name


def _parse_filter_info(filter_info):
    """
    Parse optional column(s) of input DataFrame to `build_models` and
    `apply_models`.

    Checks argument requirements.

    Parameters
    ----------
    filter_info : DataFrame
        Optional columns beyond the required columns in input DataFrame
        ``img_info_df``.

    Returns
    -------
    filter_info : DataFrame
        Entries that do not contain invalid characters.

    Raises
    ------
    ValueError
        If any entry contains prohibited character.
        Raised by `_check_prohibited_char`.

    """
    if filter_info.size == 0:
        return filter_info
    else:
        for info in filter_info:
            _check_prohibited_char(info, 'file')
        return filter_info


def initialize_model(model_name, num_points, num_clusters):
    """
    Initialize the vampire model.

    Parameters
    ----------
    model_name : str
        Name of the model.
    num_points : int
        Number of sample points of object contour.
    num_clusters : int
        Number of clusters of K-means clustering.

    Returns
    -------
    model : dict
        Contains information about the model. See notes.

    Notes
    -----
    The VAMPIRE model is contained the dict ``model``, which is stored
    in a ``.pickle`` file named by the model name. The following can
    properties can be accessed as attributes or keys.
    For more information, see :ref:`the VAMPIRE model <the_vampire_model>`.

    model_name : str
        Name of the model (build image set).
    num_point : int
        Number of points used to describe the contour.
    num_clusters : int
        Number of cluster centers for K-means clustering.
    num_pc : int
        Number of principal components used after truncation.
    mean_registered_contour : ndarray
        Mean registered contour.
    mean_aligned_contour : ndarray
        Mean aligned contour.
    principal_directions : ndarray
        Principal direction of PCA.
    centroids : ndarray
        Cluster center in K-means clustering.
    build_contours_df : DataFrame
        DataFrame of objects' contour coordinates with cluster id.

    """
    model = {
        'model_name': model_name,
        'num_points': num_points,
        'num_clusters': num_clusters,
        'num_pc': 20,  # TODO: figure out best way to deal with num pc
        'mean_registered_contour': None,
        'mean_aligned_contour': None,
        'principal_directions': None,
        'centroids': None,
        'build_contours_df': None,
    }
    return model


def build_model(img_set_path, output_path, model_name, num_points, num_clusters, filter_info, random_state=None):
    """
    Builds VAMPIRE model to one image set.

    Parameters
    ----------
    img_set_path : str
        Path to the directory containing the image set(s) used to build model.
    output_path : str
        Path of the directory used to output model and figures. Defaults to
        ``img_set_path``.
    model_name : str
        Name of the model. Defaults to time of function call.
    num_points : int
        Number of sample points of object contour. Defaults to 50.
    num_clusters : int
        Number of clusters of K-means clustering. Defaults to 5. Recommended
        range [2, 10].
    filter_info : DataFrame
        Optional columns beyond the required columns in input DataFrame
        ``img_info_df``.
    random_state : int, optional
        Random state of random processes.

    See Also
    --------
    build_models : Building multiple models using different images/conditions.

    """
    np.set_printoptions(precision=5, suppress=True)  # used for testing

    contours = extraction.extract_contours(img_set_path, filter_info)
    contours = processing.sample_contours(contours, num_points=num_points)
    contours, mean_registered_contour = processing.register_contours(contours)
    contours, mean_aligned_contour = processing.align_contours(contours, mean_registered_contour)

    principal_directions, principal_components = analysis.pca_contours(contours)
    contours_df, centroids = analysis.cluster_contours(principal_components, contours,
                                                           num_clusters=num_clusters,
                                                           num_pc=20, random_state=random_state)
    # save model parameters
    model = initialize_model(model_name, num_points, num_clusters)
    model['mean_registered_contour'] = mean_registered_contour
    model['mean_aligned_contour'] = mean_aligned_contour
    model['principal_directions'] = principal_directions
    model['build_contours_df'] = contours_df
    model['centroids'] = centroids
    util.write_pickle(os.path.join(output_path, f'{model_name}.pickle'), model)
    # print(model)

    plot.set_plot_style()
    plot.plot_distribution_contour_dendrogram(contours_df, None, output_path, model_name)
    return


def build_models(img_info_df, random_state=None):
    """
    Builds all models from the input info of image sets.

    Parameters
    ----------
    img_info_df : DataFrame
        Contains all information about image sets to be analyzed. See notes.
    random_state : int, optional
        Random state of random processes.

    Notes
    -----
    Learn more about :ref:`basics <build_basics>` and
    :ref:`advanced <build_advanced>` input requirement
    and examples. Below is a general description.

    .. rubric:: **Required columns of** ``img_info_df`` **(col 1-5)**

    The input DataFrame ``img_info_df`` must contain, *in order*, the 5
    required columns of

    img_set_path : str
        Path to the directory containing the image set(s) used to build model.
    output_path : str
        Path of the directory used to output model and figures. Defaults to
        ``img_set_path``.
    model_name : str, default
        Name of the model. Defaults to time of function call.
    num_points : int, default
        Number of sample points of object contour. Defaults to 50.
    num_clusters : int, default
        Number of clusters of K-means clustering. Defaults to 5. Recommended
        range [2, 10].

    in the first 5 columns.
    The default values are used in default columns when (1) the space is left
    blank in ``csv``/``excel`` file before converting to DataFrame, or
    (2) the space is ``None``/``np.NaN`` in the DataFrame.

    .. warning::
       The required columns must appear in order in the first 5 columns,
       even when defaults are used.


    .. rubric:: **Optional columns of** ``img_info_df`` **(col 6-)**

    The input DataFrame ``img_info_df`` could also contain any number (none
    to many) of optional columns at the right of the required columns.
    These optional columns serve as filters to the image filenames.
    The images with filenames containing values of all filters are used
    in analysis.

    filter1 : str, optional
        Unique filter of image filenames to be analyzed. E.g. "c1" for channel
        1.
    filter2 : str, optional
        Unique filter of image filenames to be analyzed. E.g. "cortex" for
        sample region.
    ... : str, optional
        Unique filter of image filenames to be analyzed. E.g. "40x" for
        magnification.

    .. tip::
       The column names of optional columns does not affect the analysis.
       The values in the columns only serves as filters to images to be
       analyzed.

    """
    _build_models_parse_df(img_info_df)
    num_img_set, num_args = img_info_df.shape

    for row_i in range(num_img_set):
        # parse arguments
        img_info = img_info_df.iloc[row_i, :]
        required_info = img_info[:5]  # 5 cols expected in doc
        filter_info = img_info[5:].values.astype(str)
        img_set_path, output_path, model_name, num_points, num_clusters = _build_models_parse_required_info(required_info)
        filter_info = _parse_filter_info(filter_info)
        # build model
        build_model(img_set_path, output_path, model_name, num_points, num_clusters, filter_info, random_state)
    return


def apply_model(img_set_path, model_path, output_path, img_set_name, filter_info):
    """
     Apply VAMPIRE model to one image set.

    Parameters
    ----------
    img_set_path : str
        Path to the directory containing the image set(s) used to apply model.
    model_path : str
        Path to the pickle file that stores model information.
    output_path : str
        Path of the directory used to output model and figures. Defaults to
        ``img_set_path``.
    img_set_name : str
        Name of the image set being applied to.
        Defaults to time of function call.
    filter_info : DataFrame
        Optional columns beyond the required columns in input DataFrame
        ``img_info_df``.

    See Also
    --------
    apply_models : Apply multiple models using different images/conditions.

    """
    # load model parameters
    model = util.read_pickle(model_path)  # type: dict
    model_name = model['model_name']
    num_points = model['num_points']
    model_mean_registered_contour = model['mean_registered_contour']
    mean_aligned_contour = model['mean_aligned_contour']
    principal_directions = model['principal_directions']
    centroids = model['centroids']
    build_contours_df = model['build_contours_df']

    contours = extraction.extract_contours(img_set_path, filter_info)
    contours = processing.sample_contours(contours, num_points=num_points)
    contours, _ = processing.register_contours(contours)
    contours, _ = processing.align_contours(contours, model_mean_registered_contour)

    principal_components = analysis.pca_transform_contours(contours, mean_aligned_contour, principal_directions)
    apply_contours_df, min_distance = analysis.assign_clusters_id(principal_components, contours, centroids, num_pc=20)
    util.write_clusters_info(img_set_path, filter_info, apply_contours_df, min_distance)

    plot.set_plot_style()
    plot.plot_distribution_contour_dendrogram(build_contours_df, apply_contours_df, output_path, model_name, img_set_name)
    return


def apply_models(img_info_df):
    """
    Applies all models from the input info of image sets.

    Parameters
    ----------
    img_info_df : DataFrame
        Contains all information about image sets to be analyzed. See notes.

    Notes
    -----
    Learn more about :ref:`basics <apply_basics>` and
    :ref:`advanced <apply_advanced>` input requirement
    and examples. Below is a general description.

    .. rubric:: **Required columns of** ``img_info_df`` **(col 1-4)**

    The input DataFrame ``img_info_df`` must contain, *in order*, the 4
    required columns of

    img_set_path : str
        Path to the directory containing the image set(s) used to apply model.
    model_path : str
        Path to the pickle file that stores model information.
    output_path : str
        Path of the directory used to output model and figures. Defaults to
        ``img_set_path``.
    img_set_name : str, default
        Name of the image set being applied to.
        Defaults to time of function call.

    in the first 4 columns.
    The default values are used in default columns when (1) the space is left
    blank in ``csv``/``excel`` file before converting to DataFrame, or
    (2) the space is ``None``/``np.NaN`` in the DataFrame.

    .. warning::
       The required columns must appear in order in the first 4 columns,
       even when defaults are used.


    .. rubric:: **Optional columns of** ``img_info_df`` **(col 5-)**

    The input DataFrame ``img_info_df`` could also contain any number (none
    to many) of optional columns at the right of the required columns.
    These optional columns serve as filters to the image filenames.
    The images with filenames containing values of all filters are used
    in analysis.

    filter1 : str, optional
        Unique filter of image filenames to be analyzed. E.g. "c1" for channel
        1.
    filter2 : str, optional
        Unique filter of image filenames to be analyzed. E.g. "cortex" for
        sample region.
    ... : str, optional
        Unique filter of image filenames to be analyzed. E.g. "40x" for
        magnification.

    .. tip::
       The column names of optional columns does not affect the analysis.
       The values in the columns only serves as filters to images to be
       analyzed.

    """
    _apply_models_parse_df(img_info_df)
    num_img_set, num_args = img_info_df.shape
    for row_i in range(num_img_set):
        # parse arguments
        img_info = img_info_df.iloc[row_i, :]
        required_info = img_info[:4]  # 4 cols expected in doc
        filter_info = img_info[4:].values.astype(str)
        img_set_path, model_path, output_path, img_set_name = _apply_models_parse_required_info(required_info)
        filter_info = _parse_filter_info(filter_info)
        # apply model
        apply_model(img_set_path, model_path, output_path, img_set_name, filter_info)
    return


def my_test():
    import time
    start = time.time()
    build_models(pd.read_csv(r'C:\Files\UniversityofWashington\_nance-lab\projects\ogd-vampire\general-pipeline\test-build.csv'), random_state=1)
    apply_models(pd.read_csv(r'C:\Files\UniversityofWashington\_nance-lab\projects\ogd-vampire\general-pipeline\test-apply.csv'))
    end = time.time()
    print(end - start)
    print('done')


# my_test()
