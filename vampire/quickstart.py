import os
from datetime import datetime

import numpy as np
import pandas as pd

from . import extraction
from . import model
from . import plot
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


def _parse_filter_info(filter_info):
    """
    Parse optional column(s) of input DataFrame to `build_models` and
    `apply_models`.

    Checks argument requirements.

    Parameters
    ----------
    filter_info : ndarray
        Regex filter(s) of image filenames to be analyzed.
        Empty if no filter needed.

    Returns
    -------
    filter_info : ndarray
        Entries that do not contain invalid characters.

    Raises
    ------
    ValueError
        If any entry contains prohibited character.
        Raised by `_check_prohibited_char`.

    """
    if filter_info.size == 0:
        return np.array([], dtype=str)
    else:
        return filter_info[filter_info != 'nan']


def _build_models_check_df(img_info_df):
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


def _build_models_check_required_info(required_info):
    """
    Parse required columns of input DataFrame to `build_models`.

    Checks argument requirements and sets default arguments.

    Parameters
    ----------
    required_info : Series
        One row of required columns (1-5) of input DataFrame
        ``img_info_df``.

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


def build_model(img_set_path, output_path,
                model_name, num_points,
                num_clusters, filter_info,
                random_state=None, savefig=True):
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
    filter_info : ndarray
        Regex filter(s) of image filenames to be analyzed.
        Empty if no filter needed.
    random_state : int, optional
        Random state of random processes.
    savefig : bool, optional
        Whether save distribution contour dendrogram.

    See Also
    --------
    build_models : Building multiple models using different images/conditions.

    """
    # get data
    properties_df = extraction.extract_properties(img_set_path,
                                                  filter_info,
                                                  write=True)
    # build model
    vampire_model = model.Vampire(model_name,
                                  num_points=num_points,
                                  num_clusters=num_clusters,
                                  num_pc=20,
                                  random_state=random_state)
    vampire_model.build(properties_df)
    # write model
    model_output_path = util.get_model_pickle_path(output_path,
                                                   filter_info,
                                                   vampire_model)
    util.write_pickle(model_output_path, vampire_model)
    # plot result
    if savefig:
        plot.set_plot_style()
        fig, axs = plot.plot_distribution_contour_dendrogram(vampire_model)
        plot.save_fig(fig,
                      output_path,
                      'shape_mode',
                      '.png',
                      model_name)
    return vampire_model


def build_models(img_info_df, random_state=None, savefig=True):
    """
    Builds all models from the input info of image sets.

    Parameters
    ----------
    img_info_df : DataFrame
        Contains all information about image sets to be analyzed. See notes.
    random_state : int, optional
        Random state of random processes.
    savefig : bool, optional
        Whether save distribution contour dendrogram.

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
        Regex filter of image filenames to be analyzed. E.g. "c1" for channel
        1.
    filter2 : str, optional
        Regex filter of image filenames to be analyzed. E.g. "cortex" for
        sample region.
    ... : str, optional
        Regex filter of image filenames to be analyzed. E.g. "40x" for
        magnification.

    .. tip::
       The column names of optional columns does not affect the analysis.
       The values in the columns only serves as filters to images to be
       analyzed.

    """
    _build_models_check_df(img_info_df)
    num_img_set, num_args = img_info_df.shape

    for row_i in range(num_img_set):
        # parse arguments
        img_info = img_info_df.iloc[row_i, :]
        required_info = img_info[:5]  # 5 cols expected in doc
        filter_info = img_info[5:].values.astype(str)
        img_set_path, \
            output_path, \
            model_name, \
            num_points, \
            num_clusters = _build_models_check_required_info(required_info)
        filter_info = _parse_filter_info(filter_info)
        # build model
        build_model(img_set_path,
                    output_path,
                    model_name,
                    num_points,
                    num_clusters,
                    filter_info,
                    random_state=random_state,
                    savefig=savefig)
    return


def _apply_models_check_df(img_info_df):
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


def _apply_models_check_required_info(required_info):
    """
    Parse required columns of input DataFrame to `apply_models`.

    Parameters
    ----------
    required_info : Series
        One row of required columns (1-4) of input DataFrame
        ``img_info_df``.

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


def apply_model(img_set_path, model_path,
                output_path, img_set_name,
                filter_info, write_csv=True,
                savefig=True):
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
    filter_info : ndarray
        Regex filter(s) of image filenames to be analyzed.
        Empty if no filter needed.
    write_csv : bool, optional
        Whether write apply model data to csv.
        Could be time consuming if csv is large.
    savefig : bool, optional
        Whether save distribution contour dendrogram.

    See Also
    --------
    apply_models : Apply multiple models using different images/conditions.

    """
    # get model
    vampire_model = util.read_pickle(model_path)
    # get data
    properties_df = extraction.extract_properties(img_set_path,
                                                  filter_info,
                                                  write=True)
    # apply model
    apply_properties_df = vampire_model.apply(properties_df)
    # write apply model data
    properties_pickle_path = util.get_apply_properties_pickle_path(output_path,
                                                                   filter_info,
                                                                   vampire_model,
                                                                   img_set_name)
    util.write_pickle(properties_pickle_path, apply_properties_df)
    # plot result
    if savefig:
        plot.set_plot_style()
        fig, axs = plot.plot_distribution_contour_dendrogram(vampire_model,
                                                             apply_properties_df)
        plot.save_fig(fig,
                      output_path,
                      'shape_mode',
                      '.png',
                      vampire_model.model_name,
                      img_set_name)

    # write apply model data to csv
    # time consuming if csv is large
    if write_csv:
        properties_csv_path = util.get_apply_properties_csv_path(output_path,
                                                                 filter_info,
                                                                 vampire_model,
                                                                 img_set_name)
        apply_properties_df.drop(['raw_contour',
                                  'normalized_contour'],
                                 axis=1) \
                           .to_csv(properties_csv_path, index=False)
    return apply_properties_df


def apply_models(img_info_df, write_csv=True, savefig=True):
    """
    Applies all models from the input info of image sets.

    Parameters
    ----------
    img_info_df : DataFrame
        Contains all information about image sets to be analyzed. See notes.
    write_csv : bool, optional
        Whether write apply model data to csv.
        Could be time consuming if csv is large.
    savefig : bool, optional
        Whether save distribution contour dendrogram.

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
        Regex filter of image filenames to be analyzed. E.g. "c1" for channel
        1.
    filter2 : str, optional
        Regex filter of image filenames to be analyzed. E.g. "cortex" for
        sample region.
    ... : str, optional
        Regex filter of image filenames to be analyzed. E.g. "40x" for
        magnification.

    .. tip::
       The column names of optional columns does not affect the analysis.
       The values in the columns only serves as filters to images to be
       analyzed.

    """
    _apply_models_check_df(img_info_df)
    num_img_set, num_args = img_info_df.shape
    for row_i in range(num_img_set):
        # parse arguments
        img_info = img_info_df.iloc[row_i, :]
        required_info = img_info[:4]  # 4 cols expected in doc
        filter_info = img_info[4:].values.astype(str)
        img_set_path, \
            model_path, \
            output_path, \
            img_set_name = _apply_models_check_required_info(required_info)
        filter_info = _parse_filter_info(filter_info)
        # apply model
        apply_model(img_set_path,
                    model_path,
                    output_path,
                    img_set_name,
                    filter_info,
                    write_csv=write_csv,
                    savefig=savefig)
    return
