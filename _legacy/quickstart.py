import os
from datetime import datetime

import numpy as np
import pandas as pd

from . import extraction, model, plot, util


def _check_char(text, input_type='path'):
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
        raise ValueError(
            'Unrecognized input_type: {input_type}'
            'Expect one in {"path", "file"}'
        )
    have_prohibited_char = any(
        prohibited_char in text
        for prohibited_char in prohibited_chars
    )
    if have_prohibited_char:
        raise ValueError(
            f'Filename-related entry of {text} contains prohibited character(s). \n'
            'Expect a model name without any of the following characters: \n'
            '\\  /  ,  :  *  "  <  >  |'
        )


def _parse_filter_info(filter_info):
    """
    Parse optional column(s) of input DataFrame to `fit_models` and
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


def fit_model(
    img_set_path,
    output_path,
    model_name,
    n_points,
    n_clusters,
    n_pcs,
    filter_info,
    write_contour=False,
    random_state=None,
    savefig=True,
):
    """
    Fits VAMPIRE model to one image set.

    Parameters
    ----------
    img_set_path : str
        Path to the directory containing the image set(s) used to fit model.
    output_path : str
        Path of the directory used to output model and figures. Defaults to
        ``img_set_path``.
    model_name : str
        Name of the model. Defaults to time of function call.
    n_points : int
        Number of sample points of object contour. Defaults to 50.
    n_clusters : int
        Number of clusters of K-means clustering. Defaults to 5. Recommended
        range [2, 10].
    n_pcs : int or None
        Number of principal components kept for analysis. Default to keeping
        those that explains 95% of total variance. Recommended to adjust
        after analyzing scree plot.
    write_contour : bool, optional
        Whether write and save raw contour coordinates.
    filter_info : ndarray
        Regex filter(s) of image filenames to be analyzed.
        Empty if no filter needed.
    random_state : int, optional
        Random state of random processes.
    savefig : bool, optional
        Whether save distribution contour dendrogram.

    See Also
    --------
    fit_models : Fitting multiple models using different images/conditions.

    """
    # get data
    properties_df = extraction.extract_properties(
        img_set_path,
        filter_info,
        write=True,
        write_contour=write_contour,
    )
    # fit model
    vampire_model = model.Vampire(
        model_name,
        n_points=n_points,
        n_clusters=n_clusters,
        n_pcs=n_pcs,
        random_state=random_state,
    )
    vampire_model.fit(properties_df)
    # write model
    model_output_path = util.get_model_pickle_path(
        output_path,
        filter_info,
        vampire_model,
    )
    util.write_pickle(model_output_path, vampire_model)
    # plot result
    if savefig:
        plot.set_plot_style()
        fig, axs = plot.plot_distribution_contour_dendrogram(vampire_model)
        plot.save_fig(
            fig,
            output_path,
            'shape_mode',
            '.png',
            model_name
        )
    return vampire_model


def fit_models(img_info_df, random_state=None, savefig=True):
    """
    Fits all models from the input info of image sets.

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
    Learn more about :ref:`basics <fit_basics>` and
    :ref:`advanced <fit_advanced>` input requirement
    and examples. Below is a general description.

    .. rubric:: **Required columns of** ``img_info_df`` **(col 1-6)**

    The input DataFrame ``img_info_df`` must contain, *in order*, the 6
    required columns of

    img_set_path : str
        Path to the directory containing the image set(s) used to fit model.
    output_path : str
        Path of the directory used to output model and figures. Defaults to
        ``img_set_path``.
    model_name : str, default
        Name of the model. Defaults to time of function call.
    n_points : int, default
        Number of sample points of object contour. Defaults to 50.
    n_clusters : int, default
        Number of clusters of K-means clustering. Defaults to 5. Recommended
        range [2, 10].
    n_pcs : int, default
        Number of principal components kept for analysis. Default to keeping
        those that explains 95% of total variance. Recommended to adjust
        after analyzing scree plot.

    in the first 5 columns.
    The default values are used in default columns when (1) the space is left
    blank in ``csv``/``excel`` file before converting to DataFrame, or
    (2) the space is ``None``/``np.NaN`` in the DataFrame.

    .. warning::
       The required columns must appear in order in the first 5 columns,
       even when defaults are used.


    .. rubric:: **Optional columns of** ``img_info_df`` **(col 7-)**

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

    def check_info_df(img_info_df):
        """
        Checks if input DataFrame to `fit_models` has the appropriate shape.

        Raises
        ------
        ValueError
            Empty DataFrame without information in rows.
        ValueError
            DataFrame does not contain required columns specified in the doc.

        """
        n_img_sets, n_args = img_info_df.shape
        if n_img_sets == 0:
            raise ValueError('Input DataFrame is empty. Expect at least one row.')
        if n_args < 6:  # 5 cols required by doc
            raise ValueError(
                'Input DataFrame does not have enough number of columns. \n'
                'Expect required 6 columns in order: img_set_path, output_path, '
                'model_name, n_points, n_clusters, n_pcs.'
            )

    def check_required_info(required_info):
        """
        Parse required columns of input DataFrame to `fit_models`.

        Checks argument requirements and sets default arguments.

        Raises
        ------
        FileNotFoundError
            If ``img_set_path`` does not exist.

        """
        # unpack args
        img_set_path, output_path, model_name, n_points, n_clusters, n_pcs = required_info

        # img_set_path
        if os.path.isdir(img_set_path):
            img_set_path = os.path.normpath(img_set_path)
        else:
            raise FileNotFoundError(
                f'Input DataFrame column 1 gives non-existing directory: \n'
                f'{img_set_path} \n'
                'Expect an existing directory with images used to fit model.'
            )

        # output_path
        if pd.isna(output_path):
            output_path = os.path.normpath(img_set_path)  # default
        else:
            _check_char(output_path)
            output_path = os.path.normpath(output_path)
            if not os.path.isdir(output_path):
                os.mkdir(output_path)

        # model_name
        if pd.isna(model_name):
            time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            model_name = time_stamp
        else:
            _check_char(model_name, 'file')

        # n_points
        if pd.isna(n_points):
            n_points = 50
        else:
            n_points = int(n_points)

        # n_clusters
        if pd.isna(n_clusters):
            n_clusters = 5
        else:
            n_clusters = int(n_clusters)

        # n_pcs
        if pd.isna(n_pcs):
            n_pcs = None
        else:
            n_pcs = int(n_pcs)

        return img_set_path, output_path, model_name, n_points, n_clusters, n_pcs

    # start of main function
    check_info_df(img_info_df)
    n_img_set, n_args = img_info_df.shape

    for row_i in range(n_img_set):
        # parse arguments
        img_info = img_info_df.iloc[row_i, :]
        required_info = img_info[:6]  # 6 cols expected in doc
        filter_info = img_info[6:].values.astype(str)
        (img_set_path,
            output_path,
            model_name,
            n_points,
            n_clusters,
            n_pcs) = check_required_info(required_info)
        filter_info = _parse_filter_info(filter_info)
        # fit model
        fit_model(
            img_set_path,
            output_path,
            model_name,
            n_points,
            n_clusters,
            n_pcs,
            filter_info,
            random_state=random_state,
            savefig=savefig
        )
    return


def transform_dataset(
    img_set_path,
    model_path,
    output_path,
    img_set_name,
    filter_info,
    write_csv=True,
    write_contour=False,
    savefig=True,
):
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
        Could be time-consuming if csv is large.
    write_contour : bool, optional
        Whether write and save raw contour coordinates.
    savefig : bool, optional
        Whether save distribution contour dendrogram.

    See Also
    --------
    apply_models : Apply multiple models using different images/conditions.

    """
    # get model
    vampire_model = util.read_pickle(model_path)
    # get data
    properties_df = extraction.extract_properties(
        img_set_path,
        filter_info,
        write=True,
        write_contour=write_contour,
    )
    # apply model
    apply_properties_df = vampire_model.transform(properties_df)
    # write apply model data
    properties_pickle_path = util.get_apply_properties_pickle_path(
        output_path,
        filter_info,
        vampire_model,
        img_set_name
    )
    util.write_pickle(properties_pickle_path, apply_properties_df)
    # plot result
    if savefig:
        plot.set_plot_style()
        fig, axs = plot.plot_distribution_contour_dendrogram(
            vampire_model,
            apply_properties_df
        )
        plot.save_fig(
            fig,
            output_path,
            'shape_mode',
            '.png',
            vampire_model.model_name,
            img_set_name
        )

    # write apply model data to csv
    # time-consuming if csv is large
    if write_csv:
        properties_csv_path = util.get_apply_properties_csv_path(
            output_path,
            filter_info,
            vampire_model,
            img_set_name
        )
        apply_properties_df.drop(
            ['raw_contour', 'normalized_contour'],
            axis=1
        ).to_csv(
            properties_csv_path,
            index=False
        )
    return apply_properties_df


def transform_datasets(img_info_df, write_csv=True, savefig=True):
    """
    Applies all models from the input info of image sets.

    Parameters
    ----------
    img_info_df : DataFrame
        Contains all information about image sets to be analyzed. See notes.
    write_csv : bool, optional
        Whether write transformed data to csv.
        Could be time-consuming if csv is large.
    savefig : bool, optional
        Whether save distribution contour dendrogram.

    Notes
    -----
    Learn more about :ref:`basics <transform_basics>` and
    :ref:`advanced <transform_advanced>` input requirement
    and examples. Below is a general description.

    .. rubric:: **Required columns of** ``img_info_df`` **(col 1-4)**

    The input DataFrame ``img_info_df`` must contain, *in order*, the 4
    required columns of

    img_set_path : str
        Path to the directory containing the image set(s) used to transform data.
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

    def check_info_df(img_info_df):
        """
        Checks if input DataFrame to `apply_models` has the appropriate shape.

        Raises
        ------
        ValueError
            Empty DataFrame without information in rows.
        ValueError
            DataFrame does not contain required columns specified in the doc.

        """
        n_img_set, n_args = img_info_df.shape
        if n_img_set == 0:
            raise ValueError('Input DataFrame is empty. Expect at least one row.')
        if n_args < 4:  # 4 cols required by doc
            raise ValueError(
                'Input DataFrame does not have enough number of columns. \n'
                'Expect required 3 columns in order: img_set_path, model_path, '
                'output_path.'
            )
        return

    def check_required_info(required_info):
        """
        Parse required columns of input DataFrame to `apply_models`.

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
            raise FileNotFoundError(
                f'Input DataFrame column 1 gives non-existing directory: \n'
                f'{img_set_path} \n'
                'Expect an existing directory with images used to apply model.'
            )

        # model_path
        if os.path.isfile(model_path):
            filename, extension = os.path.splitext(model_path)
            if extension == '.pickle':
                model_path = os.path.normpath(model_path)
            else:
                raise ValueError(
                    f'Input DataFrame column 2 gives non-pickle file: \n'
                    f'{model_path} \n'
                    'Expect an existing pickle file for model information.'
                )
        else:
            raise FileNotFoundError(
                f'Input DataFrame column 2 gives non-existing file: \n'
                f'{model_path} \n'
                'Expect an existing pickle file for model information.'
            )

        # output_path
        if pd.isna(output_path):
            output_path = os.path.normpath(img_set_path)
        else:
            _check_char(output_path)
            output_path = os.path.normpath(output_path)
            if not os.path.isdir(output_path):
                os.mkdir(output_path)

        # img_set_name
        if pd.isna(img_set_name):
            time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            img_set_name = time_stamp
        else:
            _check_char(img_set_name, 'file')

        return img_set_path, model_path, output_path, img_set_name

    # start of main function
    check_info_df(img_info_df)
    n_img_set, n_args = img_info_df.shape
    for row_i in range(n_img_set):
        # parse arguments
        img_info = img_info_df.iloc[row_i, :]
        required_info = img_info[:4]  # 4 cols expected in doc
        filter_info = img_info[4:].values.astype(str)
        (img_set_path,
            model_path,
            output_path,
            img_set_name) = check_required_info(required_info)
        filter_info = _parse_filter_info(filter_info)
        # transform data
        transform_dataset(
            img_set_path,
            model_path,
            output_path,
            img_set_name,
            filter_info,
            write_csv=write_csv,
            savefig=savefig
        )
    return
