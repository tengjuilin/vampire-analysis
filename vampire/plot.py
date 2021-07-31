import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy import cluster, spatial


def _get_df_info(contours_df):
    """
    Returns mean of DataFrame columns by cluster id and number of
    clusters, coordinates, and points.

    Parameters
    ----------
    contours_df : DataFrame
        DataFrame containing contour coordinates of objects and a column
        named 'cluster_id' that indicates the cluster that an object belongs.

    Returns
    -------
    mean_contours : DataFrame
        Mean of columns (contours) by cluster id.
    num_clusters : int
        Number of clusters.
    num_coords : int
        Number of contour coordinates.
    num_points : int
        Number of points in a contour.

    """
    mean_contours = contours_df.groupby('cluster_id').mean().values
    num_clusters, num_coords = mean_contours.shape
    if num_coords % 2:
        raise ValueError('Coordinate does not have matching number of x and y coordinates.')
    num_points = num_coords // 2
    return mean_contours, num_clusters, num_coords, num_points


def set_plot_style():
    """
    Set matplotlib plot settings of ``rcParams`` for better visualization.

    References
    ----------
    [1] Customizing Matplotlib with style sheets and rcParams.
    https://matplotlib.org/stable/tutorials/introductory/customizing.html

    """
    plt.rcParams.update({
        'font.family': 'Arial',  # Times New Roman, Calibri
        'font.weight': 'normal',
        'mathtext.fontset': 'cm',
        'font.size': 18,

        'lines.linewidth': 2,

        'axes.linewidth': 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titleweight': 'bold',
        'axes.titlesize': 18,
        'axes.labelweight': 'bold',

        'xtick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.size': 8,
        'ytick.major.width': 2,

        'figure.dpi': 80,
        'savefig.dpi': 300,

        'legend.framealpha': 1,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        'legend.fontsize': 14,
    })


def save_fig(fig, output_path, fig_type, build_name=None, apply_name=None):
    """
    Save figure to local directory.

    Parameters
    ----------
    fig : Figure
        Figure to be saved.
    output_path : str
        Path to the output directory.
    fig_type : str
        General description/type of the figure.
    build_name : str, optional
        Name of the built model.
    apply_name : str, optional
        Name of the image set being applied to.

    """
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if build_name is None:
        fig_path = os.path.join(output_path, f'{fig_type}_{time_stamp}.png')
    else:
        if apply_name is None:  # build model
            fig_path = os.path.join(output_path, f'{fig_type}_build_{build_name}.png')
        else:  # apply model
            fig_path = os.path.join(output_path, f'{fig_type}_apply_{build_name}_on_{apply_name}.png')

    if os.path.exists(fig_path):
        fig_name, extension = os.path.splitext(fig_path)
        fig.savefig(f'{fig_name}_{time_stamp}{extension}', dpi=300)
    else:
        fig.savefig(fig_path, dpi=300)
    return


def plot_dendrogram(contours_df,
                    output_path=None, build_name=None, apply_name=None,
                    ax=None, fig_size=(6, 2)):
    """
    Plots dendrogram of mean contours.

    Parameters
    ----------
    contours_df : DataFrame
        DataFrame containing contour coordinates of objects and a column
        named 'cluster_id' that indicates the cluster that an object belongs.
    output_path : str, optional
        Path to the output directory. Default None, does not save figure.
        Cannot be not None at the same time with ``ax``.
    build_name : str, optional
        Name of the built model.
    apply_name : str, optional
        Name of the image set being applied to.
    ax : Axes, optional
        Figure axis to be plotted on. Cannot be not None with ``output_path``
        at the same time.
    fig_size : (float, float), optional
        Width, height in inches. Default (6, 2).

    Returns
    -------
    object_index : ndarray of str
        The cluster that each branch represents.

    See Also
    --------
    scipy.cluster.hierarchy.linkage, scipy.cluster.hierarchy.dendrogram

    """
    mean_contours, num_clusters, num_coords, num_points = _get_df_info(contours_df)
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        if output_path is not None:
            raise ValueError('Unable to save figure. output_path and ax cannot be not None at the same time.')

    pair_distance = spatial.distance.pdist(mean_contours, 'euclidean')
    linkage_mat = cluster.hierarchy.linkage(pair_distance, method='complete')
    linkage_mat[:, 2] = linkage_mat[:, 2] * 5  # scale distance for less cluttered visualization

    # plot dendrogram
    cluster.hierarchy.set_link_color_palette(['k'])
    branches = cluster.hierarchy.dendrogram(linkage_mat, ax=ax, p=0,
                                            truncate_mode='mlab',
                                            orientation='bottom',
                                            above_threshold_color='k')
    object_index = np.array(branches['ivl'], dtype=int)  # make connection between item and dendrogram
    ax.axis('off')

    # save figure
    if output_path is not None:
        save_fig(fig, output_path, 'dendrogram', build_name, apply_name)
    return object_index


def plot_contours(contours_df, object_index,
                  output_path=None, build_name=None, apply_name=None, contour_scale=3,
                  ax=None, fig_size=(6, 2), color=None, alpha=None, linewidth=None):
    """
    Plots mean contours.

    Parameters
    ----------
    contours_df : DataFrame
        DataFrame containing contour coordinates of objects and a column
        named 'cluster_id' that indicates the cluster that an object belongs.
    object_index : ndarray of str
        A list of labels corresponding to the leaf nodes.
    output_path : str, optional
        Path to the output directory. Default None, does not save figure.
        Cannot be not None at the same time with ``ax``.
    build_name : str, optional
        Name of the built model.
    apply_name : str, optional
        Name of the image set being applied to.
    contour_scale : float, optional
        Scale of the contour shapes.
    ax : Axes, optional
        Figure axis to be plotted on. Cannot be not None with ``output_path``
        at the same time.
    fig_size : (float, float), optional
        Width, height in inches. Default (6, 2).
    color : str, optional
        Color of mean contours.
    alpha : float, optional
        Alpha of mean contours.
    linewidth : float, optional
        Line width of mean contours.

    """
    mean_contours, num_clusters, num_coords, num_points = _get_df_info(contours_df)
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        if output_path is not None:
            raise ValueError('Unable to save figure. output_path and ax cannot be not None at the same time.')

    num_clusters, num_coords = mean_contours.shape
    num_points = num_coords // 2

    x_first = 5  # offset of first contour
    x_offset = 10  # offset between contours
    for i in range(num_clusters):
        # read in contour coordinates
        x = mean_contours[object_index[i], :num_points]
        y = mean_contours[object_index[i], num_points:]
        # form close shape when plotting
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        # place shape into right location
        x = x * contour_scale + x_first + x_offset * i
        y = y * contour_scale
        # plot shape of objects corresponding to the branches
        ax.plot(x, y, color=color, alpha=alpha, lw=linewidth)
    ax.axis('equal')
    ax.axis('off')

    # save figure
    if output_path is not None:
        save_fig(fig, output_path, 'contours', build_name, apply_name)
    return


def plot_representatives(contours_df, object_index,
                         output_path=None, build_name=None, apply_name=None,
                         num_sample=10, random_state=None,
                         ax=None, fig_size=(17, 2), color=None, alpha=None, linewidth=None):
    """
    Plots representative object contours.

    Parameters
    ----------
    contours_df : DataFrame
        DataFrame containing contour coordinates of objects and a column
        named 'cluster_id' that indicates the cluster that an object belongs.
    object_index : ndarray of str
        A list of labels corresponding to the leaf nodes.
    output_path : str, optional
        Path to the output directory. Default None, does not save figure.
        Cannot be not None at the same time with ``ax``.
    build_name : str, optional
        Name of the built model.
    apply_name : str, optional
        Name of the image set being applied to.
    num_sample : int, optional
        Number of sample drawn from each cluster. Default 10. If num_sample >
        number of total available samples in the smallest cluster, it is
        set to the that number.
    random_state : int, optional
        Random state of sampling representative contours.
    ax : Axes, optional
        Figure axis to be plotted on. Cannot be not None with ``output_path``
        at the same time.
    fig_size : (float, float), optional
        Width, height in inches. Default (17, 2).
    color : str, optional
        Color of representative contours.
    alpha : float, optional
        Alpha of representative contours.
    linewidth : float, optional
        Line width of representative contours.

    """

    mean_contours, num_clusters, num_coords, num_points = _get_df_info(contours_df)
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        if output_path is not None:
            raise ValueError('Unable to save figure. output_path and ax cannot be not None at the same time.')

    x_offset = 5  # move center of another cluster to new location

    # determine sample size
    cluster_id = contours_df['cluster_id'].values
    unique, counts = np.unique(cluster_id, return_counts=True)
    max_num_sample = np.min(counts)
    if num_sample > max_num_sample:
        num_sample = max_num_sample
    # sample contours from all clusters
    all_cluster_samples_df = contours_df.groupby('cluster_id').sample(n=num_sample,
                                                                      random_state=random_state)
    sorting_index = object_index.astype(int)

    # plotting each sample contour
    for cluster_i in range(num_clusters):
        # sample contour in current cluster
        cluster_samples_df = all_cluster_samples_df[all_cluster_samples_df['cluster_id'] == sorting_index[cluster_i]]
        cluster_samples = cluster_samples_df.drop(columns='cluster_id').values
        for sample_i in range(num_sample):
            # plot each sample contour by...
            # read in sample contour coordinates
            x = cluster_samples[sample_i, :num_points]
            y = cluster_samples[sample_i, num_points:]
            # form close shape when plotting
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            # place shape into right location
            x = x + x_offset * cluster_i
            y = y
            # sample shape of objects corresponding to the clusters
            ax.plot(x, y, color=color, alpha=alpha, lw=linewidth)
    plt.axis('equal')
    plt.axis('off')

    # save figure
    if output_path is not None:
        save_fig(fig, output_path, 'representatives', build_name, apply_name)
    return


def plot_distribution(contours_df, object_index,
                      output_path=None, build_name=None, apply_name=None,
                      ax=None):
    """
    Plots the distribution of mean contours in a bar graph.

    Parameters
    ----------
    contours_df : DataFrame
        DataFrame containing contour coordinates of objects and a column
        named 'cluster_id' that indicates the cluster that an object belongs.
    object_index : ndarray of str
        A list of labels corresponding to the leaf nodes.
    output_path : str, optional
        Path to the output directory. Default None, does not save figure.
        Cannot be not None at the same time with ``ax``.
    build_name : str, optional
        Name of the built model.
    apply_name : str, optional
        Name of the image set being applied to.
    ax : Axes, optional
        Figure axis to be plotted on. Cannot be not None with ``output_path``
        at the same time.

    """
    mean_contours, num_clusters, num_coords, num_points = _get_df_info(contours_df)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        if output_path is not None:
            raise ValueError('Unable to save figure. output_path and ax cannot be not None at the same time.')

    x_first = 5  # offset of first contour
    x_offset = 10  # offset between contours

    cluster_id = contours_df['cluster_id'].values
    unique, counts = np.unique(cluster_id, return_counts=True)
    # sort the distributions corresponding to the dendrogram
    sorting_index = object_index.astype(int)
    distribution = counts / np.sum(counts) * 100  # unit: percent
    distribution = distribution[sorting_index]

    # plot shape mode  distribution bar plot
    width = x_offset / 2
    x = np.arange(x_first, num_clusters * x_offset + x_offset / 2, x_offset)
    colors = [plt.get_cmap('tab10')(cluster_i) for cluster_i in range(10)] * 5  # cycle through the 10 colors
    ax.bar(x=x, height=distribution,
           color=colors, alpha=0.7,
           align='center', width=width)
    ax.set_ylabel(f'Distribution [%]')
    ax.set_xticks([])  # clear x tick and tick labels

    # figure settings after plotting
    plt.tight_layout()

    # save figure
    if output_path is not None:
        save_fig(fig, output_path, 'distribution', build_name, apply_name)
    return


def plot_contour_dendrogram(contours_df,
                            output_path=None, build_name=None, apply_name=None,
                            fig_size=(6, 2), contour_scale=3):
    """
    Plots dendrogram with mean contours.

    Parameters
    ----------
    contours_df : DataFrame
        DataFrame containing contour coordinates of objects and a column
        named 'cluster_id' that indicates the cluster that an object belongs.
    output_path : str, optional
        Path to the output directory. Default None, does not save figure.
        Cannot be not None at the same time with ``ax``.
    build_name : str, optional
        Name of the built model.
    apply_name : str, optional
        Name of the image set being applied to.
    fig_size : (float, float), optional
        Width, height in inches. Default (6, 2).
    contour_scale : float, optional
        Scale of the contour shapes.

    Returns
    -------
    object_index : ndarray of str
        The cluster that each branch represents.

    See Also
    --------
    plot_dendrogram, plot_contours

    """
    # figure structure
    fig, axs = plt.subplots(2, 1, figsize=fig_size, frameon=False, sharex='all')

    # plot dendrogram and contour
    object_index = plot_dendrogram(contours_df, ax=axs[1])
    plot_contours(contours_df, object_index, ax=axs[0], contour_scale=contour_scale)

    # save figure
    if output_path is not None:
        save_fig(fig, output_path, 'distribution', build_name, apply_name)
    return object_index


def plot_distribution_contour(contours_df,
                              output_path=None, build_name=None, apply_name=None,
                              fig_size=(5, 5), height_ratio=(4, 1), contour_scale=3):
    """
    Plots the distribution of mean contours in a bar graph with labeling
    of mean contours.

    Parameters
    ----------
    contours_df : DataFrame
        DataFrame containing contour coordinates of objects and a column
        named 'cluster_id' that indicates the cluster that an object belongs.
    output_path : str, optional
        Path to the output directory. Default None, does not save figure.
        Cannot be not None at the same time with ``ax``.
    build_name : str, optional
        Name of the built model.
    apply_name : str, optional
        Name of the image set being applied to.
    fig_size : (float, float), optional
        Width, height in inches. Default (5, 5).
    height_ratio : list[float], optional
        Ratio between height of distribution plot, shape mode contours, and
        shape mode dendrogram. Default [4, 1, 1]. Recommended values:

            * [4, 1, 1] for 5 clusters (shape modes)
            * [4, 0.5, 1] for 10 clusters (shape modes)
            * [4, 0.35, 1] for 15 clusters (shape modes)
    contour_scale : float, optional
        Scale of the contour shapes.

    See Also
    --------
    plot_contours, plot_distribution

    """
    # figure structure
    fig, axs = plt.subplots(2, 1, figsize=fig_size, frameon=False, sharex='all',
                            gridspec_kw={'height_ratios': height_ratio})

    # plot dendrogram and contour
    object_index = plot_dendrogram(contours_df)
    plt.close()
    plot_contours(contours_df, object_index, ax=axs[1], contour_scale=contour_scale)
    plot_distribution(contours_df, object_index, ax=axs[0])

    # figure settings after plotting
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)

    # save figure
    if output_path is not None:
        save_fig(fig, output_path, 'distribution', build_name, apply_name)
    return


def plot_distribution_contour_dendrogram(build_contours_df, apply_contours_df=None,
                                         output_path=None, build_name=None, apply_name=None,
                                         fig_size=(5, 5), height_ratio=(4, 1, 1), contour_scale=3):
    """
    Plots the distribution of mean contours in a bar graph with labeling
    of mean contours and their dendrogram.

    Parameters
    ----------
    build_contours_df : DataFrame
        DataFrame containing contour coordinates of objects used to build model
        and a column named 'cluster_id' that indicates the cluster that an
        object belongs.
    apply_contours_df : DataFrame
        DataFrame containing contour coordinates of objects used to apply model
        and a column named 'cluster_id' that indicates the cluster that an
        object belongs.
    output_path : str, optional
        Path to the output directory. Default None, does not save figure.
        Cannot be not None at the same time with ``ax``.
    build_name : str, optional
        Name of the built model.
    apply_name : str, optional
        Name of the image set being applied to.
    fig_size : (float, float), optional
        Width, height in inches. Default (6, 2).
    height_ratio : list[float], optional
        Ratio between height of distribution plot, shape mode contours, and
        shape mode dendrogram. Default [4, 1, 1]. Recommended values:

            * [4, 1, 1] for 5 clusters (shape modes)
            * [4, 0.5, 1] for 10 clusters (shape modes)
            * [4, 0.35, 1] for 15 clusters (shape modes)
    contour_scale : float, optional
        Scale of the contour shapes.

    See Also
    --------
    plot_dendrogram, plot_contours, plot_distribution

    """
    # figure structure
    height_ratio = list(height_ratio)  # prevent mutable default argument
    fig, axs = plt.subplots(3, 1, figsize=fig_size, frameon=False, sharex='all',
                            gridspec_kw={'height_ratios': height_ratio})

    if apply_contours_df is None:  # build model plot
        object_index = plot_dendrogram(build_contours_df, ax=axs[2])
        plot_contours(build_contours_df, object_index, ax=axs[1], contour_scale=contour_scale)
        plot_distribution(build_contours_df, object_index, ax=axs[0])
    elif not apply_contours_df.empty:  # apply model plot
        object_index = plot_dendrogram(build_contours_df, ax=axs[2])
        plot_contours(build_contours_df, object_index, ax=axs[1], contour_scale=contour_scale,
                      color='black', alpha=0.5, linewidth=2)
        plot_contours(apply_contours_df, object_index, ax=axs[1], contour_scale=contour_scale,
                      alpha=0.7, linewidth=2)
        plot_distribution(apply_contours_df, object_index, ax=axs[0])
    else:
        raise ValueError('Got an empty DataFrame apply_contours_df. Expect None or non-empty DataFrame.')

    # figure settings after plotting
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)

    # save figure
    if output_path is not None:
        save_fig(fig, output_path, 'shape_mode', build_name, apply_name)
    return


# def plot_dendrogram(contours_df,
#                     output_path=None, build_name=None, apply_name=None,
#                     fig_size=(6, 2), contour_scale=3):
#     """
#     Plots shape mode contours and shape mode dendrogram in one figure.
#
#     Used in conjunction with shape mode distribution plot via
#     `plot_distribution`.
#
#     Parameters
#     ----------
#     contours_df : DataFrame
#         DataFrame containing contour coordinates of objects and a column
#         named 'cluster_id' that indicates the cluster that an object belongs.
#     output_path : str, optional
#         Path to the output directory. Default None, does not save figure.
#     build_name : str, optional
#         Name of the built model.
#     apply_name : str, optional
#         Name of the image set being applied to.
#     fig_size : (float, float), optional
#         Width, height in inches. Default (6, 2).
#     contour_scale : float, optional
#         Scale of contour size. Default 3. Recommended range [2, 3].
#
#     Returns
#     -------
#     fig : Figure
#     ax : array of Axes
#     object_index : ndarray of str
#         A list of labels corresponding to the leaf nodes.
#
#     See Also
#     --------
#     scipy.cluster.hierarchy.linkage : Computes the linkage matrix.
#     scipy.cluster.hierarchy.dendrogram : Plots the dendrogram.
#     plot_distribution : Plots shape mode distribution.
#
#     """
#     # testing code below
#     # mean_contours = bdst0
#     # plot_dendrogram(Z, mean_contours)
#
#     # calculate mean and useful numbers
#     mean_contours = contours_df.groupby('cluster_id').mean().values
#     num_clusters, num_coords = mean_contours.shape
#     if num_coords % 2:
#         raise ValueError('Coordinate does not have matching number of x and y coordinates.')
#     num_points = num_coords // 2
#
#     # figure structure
#     fig, axs = plt.subplots(2, 1, figsize=fig_size, frameon=False, sharex='all')
#
#     ##########################################
#     #     plot dendrogram (bottom) axs[1]    #
#     ##########################################
#     # calculate dendrogram-required linkage matrix
#     pair_distance = spatial.distance.pdist(mean_contours, 'euclidean')
#     linkage_mat = cluster.hierarchy.linkage(pair_distance, method='complete')
#     linkage_mat[:, 2] = linkage_mat[:, 2] * 5  # scale distance for less cluttered visualization
#
#     # plot dendrogram
#     cluster.hierarchy.set_link_color_palette(['k'])
#     branches = cluster.hierarchy.dendrogram(linkage_mat, ax=axs[1], p=0,
#                                             truncate_mode='mlab',
#                                             orientation='bottom',
#                                             above_threshold_color='k')
#     object_index = np.array(branches['ivl'], dtype=int)  # make connection between item and dendrogram
#     axs[1].axis('off')
#
#     ##########################################
#     # plot contours of objects (mid) axs[0]  #
#     ##########################################
#     # plot each object
#     x_first = 5  # offset of first item
#     x_offset = 10  # offset between items
#     y_offset = -5
#     for i in range(num_clusters):
#         # read in contour coordinates
#         x = mean_contours[object_index[i], :num_points]
#         y = mean_contours[object_index[i], num_points:]
#         # form close shape when plotting
#         x = np.append(x, x[0])
#         y = np.append(y, y[0])
#         # place shape into right location
#         x = x * contour_scale + x_first + x_offset * i
#         y = y * contour_scale + y_offset
#         # shape of objects corresponding to the branches
#         axs[0].plot(x, y)
#     axs[0].axis('equal')
#     axs[0].axis('off')
#
#     # save figure
#     if output_path is not None:
#         save_fig(fig, output_path, 'dendrogram', build_name, apply_name)
#     return fig, axs, object_index
#
#
# def plot_representatives(contours_df, object_index,
#                          output_path=None, build_name=None, apply_name=None,
#                          fig_size=(17, 2), num_sample=10, random_state=None):
#     """
#     Plots representative object contours.
#
#     Parameters
#     ----------
#     contours_df : DataFrame
#         DataFrame containing contour coordinates of objects and a column
#         named 'cluster_id' that indicates the cluster that an object belongs.
#     object_index : ndarray of str
#         A list of labels corresponding to the leaf nodes.
#     output_path : str, optional
#         Path to the output directory. Default None, does not save figure.
#     build_name : str, optional
#         Name of the built model.
#     apply_name : str, optional
#         Name of the image set being applied to.
#     fig_size : (float, float), optional
#         Width, height in inches. Default (17, 2).
#     num_sample : int, optional
#         Number of sample drawn from each cluster. Default 10. If num_sample >
#         number of total available samples in the smallest cluster, it is
#         set to the that number.
#     random_state : int, optional
#         Random state of sampling representative contours.
#
#     Returns
#     -------
#     fig : Figure
#     ax : array of Axes
#
#     See Also
#     --------
#     plot_dendrogram : Representative shapes should resemble the mean shape mode.
#
#     """
#     # testing code below
#     # contours_df = pd.DataFrame(bdpc)
#     # contours_df['cluster_id'] = IDX
#     # plot_representatives(contours_df)
#
#     # calculate mean and useful numbers
#     mean_contours = contours_df.groupby('cluster_id').mean().values
#     num_clusters, num_coords = mean_contours.shape
#     if num_coords % 2:
#         raise ValueError('Coordinate does not have matching number of x and y coordinates.')
#     num_points = num_coords // 2
#     x_offset = 5  # move center of another cluster to new location
#
#     # determine sample size
#     cluster_id = contours_df['cluster_id'].values
#     unique, counts = np.unique(cluster_id, return_counts=True)
#     max_num_sample = np.min(counts)
#     if num_sample > max_num_sample:
#         num_sample = max_num_sample
#     # sample contours from all clusters
#     all_cluster_samples_df = contours_df.groupby('cluster_id').sample(n=num_sample,
#                                                                       random_state=random_state)
#     sorting_index = object_index.astype(int)
#
#     # plotting each sample contour
#     fig, ax = plt.subplots(figsize=fig_size)
#     for cluster_i in range(num_clusters):
#         # sample contour in current cluster
#         cluster_samples_df = all_cluster_samples_df[all_cluster_samples_df['cluster_id'] == sorting_index[cluster_i]]
#         cluster_samples = cluster_samples_df.drop(columns='cluster_id').values
#         for sample_i in range(num_sample):
#             # plot each sample contour by...
#             # read in sample contour coordinates
#             x = cluster_samples[sample_i, :num_points]
#             y = cluster_samples[sample_i, num_points:]
#             # form close shape when plotting
#             x = np.append(x, x[0])
#             y = np.append(y, y[0])
#             # place shape into right location
#             x = x + x_offset * cluster_i
#             y = y
#             # sample shape of objects corresponding to the clusters
#             ax.plot(x, y, 'r', alpha=0.5)
#     plt.axis('equal')
#     plt.axis('off')
#
#     # save figure
#     if output_path is not None:
#         save_fig(fig, output_path, 'representatives', build_name, apply_name)
#     return fig, ax
#
#
# def plot_distribution(contours_df, object_index,
#                       output_path=None, build_name=None, apply_name=None,
#                       fig_size=(6, 4)):
#     """
#     Plots shape mode distribution bar graph.
#
#     Used in conjunction with shape mode dendrogram via `plot_dendrogram`.
#
#     Parameters
#     ----------
#     contours_df : DataFrame
#         DataFrame containing contour coordinates of objects and a column
#         named 'cluster_id' that indicates the cluster that an object belongs.
#     object_index : ndarray of str
#         A list of labels corresponding to the leaf nodes.
#     output_path : str, optional
#         Path to the output directory. Default None, does not save figure.
#     build_name : str, optional
#         Name of the built model.
#     apply_name : str, optional
#         Name of the image set being applied to.
#     fig_size : (float, float), optional
#         Width, height in inches. Default (6, 4).
#
#     Returns
#     -------
#     fig : Figure
#     ax : array of Axes
#
#     See Also
#     --------
#     plot_dendrogram : Plots shape mode contours and shape mode dendrogram.
#
#     """
#     # testing code below
#     # import pandas as pd
#     # contours_df = pd.DataFrame(bdpc)
#     # contours_df['cluster_id'] = IDX
#     # object_index = dendidx
#
#     # calculate distribution from counts
#     cluster_id = contours_df['cluster_id'].values
#     unique, counts = np.unique(cluster_id, return_counts=True)
#     num_clusters = unique.size
#     # sort the distributions corresponding to the dendrogram
#     sorting_index = object_index.astype(int)
#     distribution = counts / np.sum(counts) * 100  # unit: percent
#     distribution = distribution[sorting_index]
#
#     x_first = 5
#     x_offset = 10
#     width = x_offset / 2
#
#     # plot shape mode distribution bar plot
#     fig, ax = plt.subplots(figsize=fig_size)
#     # x = np.arange(num_clusters).astype(str)
#     x = np.arange(x_first, num_clusters * x_offset + x_offset / 2, x_offset)
#     colors = [plt.get_cmap('tab10')(cluster_i) for cluster_i in range(10)] * 5  # cycle through the 10 colors
#     ax.bar(x=x, height=distribution,
#                color=colors, alpha=0.7,
#                align='center', width=width)
#     # ax.bar(x=x, height=distribution, align='center')
#     ax.set_xlabel('Shape mode')
#     ax.set_ylabel('Distribution [%]')
#     plt.xticks(x, labels=[''] * num_clusters)  # clear x tick labels
#     plt.tight_layout()
#
#     # save figure
#     if output_path is not None:
#         save_fig(fig, output_path, 'distribution', build_name, apply_name)
#     return fig, ax
#
#
# def plot_full_old(contours_df,
#               output_path=None, build_name=None, apply_name=None,
#               fig_size=(5, 5), height_ratio=(4, 1, 1), contour_scale=3):
#     """
#     Plots shape mode distribution, mean contours, and dendrogram in one figure.
#
#     Parameters
#     ----------
#     contours_df : DataFrame
#         DataFrame containing contour coordinates of objects and a column
#         named 'cluster_id' that indicates the cluster that an object belongs.
#     output_path : str, optional
#         Path to the output directory. Default None, does not save figure.
#     build_name : str, optional
#         Name of the built model.
#     apply_name : str, optional
#         Name of the image set being applied to.
#     fig_size : (float, float), optional
#         Width, height in inches. Default (5, 5).
#     height_ratio : list[float], optional
#         Ratio between height of distribution plot, shape mode contours, and
#         shape mode dendrogram. Default [4, 1, 1]. Recommended values:
#
#             * [4, 1, 1] for 5 clusters (shape modes)
#             * [4, 0.5, 1] for 10 clusters (shape modes)
#             * [4, 0.35, 1] for 15 clusters (shape modes)
#
#     contour_scale : float, optional
#         Scale of contour size. Default 3. Recommended range [2, 3].
#
#     Returns
#     -------
#     fig : Figure
#     ax : array of Axes
#
#     See Also
#     --------
#     plot_distribution : Plots shape mode distribution.
#     plot_dendrogram : Plots shape mode contours and dendrogram in one figure.
#
#     """
#     # calculate mean and useful numbers
#     height_ratio = list(height_ratio)  # prevent mutable default argument
#     mean_contours = contours_df.groupby('cluster_id').mean().values
#     num_clusters, num_coords = mean_contours.shape
#     if num_coords % 2:
#         raise ValueError('Coordinate does not have matching number of x and y coordinates.')
#     num_points = num_coords // 2
#
#     # figure structure
#     fig, axs = plt.subplots(3, 1, figsize=fig_size, frameon=False, sharex='all',
#                             gridspec_kw={'height_ratios': height_ratio})
#
#     ##########################################
#     #     plot dendrogram (bottom) axs[2]    #
#     ##########################################
#     # calculate dendrogram-required linkage matrix
#     # contours = contours_df.drop(columns='cluster_id').values
#     # contours = mean_contours
#     pair_distance = spatial.distance.pdist(mean_contours, 'euclidean')
#     linkage_mat = cluster.hierarchy.linkage(pair_distance, method='complete')
#     linkage_mat[:, 2] = linkage_mat[:, 2] * 5  # scale distance for less cluttered visualization
#
#     # plot dendrogram
#     cluster.hierarchy.set_link_color_palette(['k'])
#     branches = cluster.hierarchy.dendrogram(linkage_mat, ax=axs[2], p=0,
#                                             truncate_mode='mlab',
#                                             orientation='bottom',
#                                             above_threshold_color='k')
#     object_index = np.array(branches['ivl'], dtype=int)  # make connection between item and dendrogram
#     axs[2].axis('off')
#
#     ##########################################
#     # plot contours of objects (mid) axs[1]  #
#     ##########################################
#     x_first = 5  # offset of first contour
#     x_offset = 10  # offset between contours
#     for i in range(num_clusters):
#         # read in contour coordinates
#         x = mean_contours[object_index[i], :num_points]
#         y = mean_contours[object_index[i], num_points:]
#         # form close shape when plotting
#         x = np.append(x, x[0])
#         y = np.append(y, y[0])
#         # place shape into right location
#         x = x * contour_scale + x_first + x_offset * i
#         y = y * contour_scale
#         # plot shape of objects corresponding to the branches
#         axs[1].plot(x, y)
#     axs[1].axis('equal')
#     axs[1].axis('off')
#
#     ##########################################
#     #   distribution bar plot (top) axs[0]   #
#     ##########################################
#     # calculate distribution from counts
#     cluster_id = contours_df['cluster_id'].values
#     unique, counts = np.unique(cluster_id, return_counts=True)
#     # sort the distributions corresponding to the dendrogram
#     sorting_index = object_index.astype(int)
#     distribution = counts / np.sum(counts) * 100  # unit: percent
#     distribution = distribution[sorting_index]
#
#     # plot shape mode  distribution bar plot
#     width = x_offset / 2
#     x = np.arange(x_first, num_clusters*x_offset + x_offset/2, x_offset)
#     colors = [plt.get_cmap('tab10')(cluster_i) for cluster_i in range(10)] * 5  # cycle through the 10 colors
#     axs[0].bar(x=x, height=distribution,
#                color=colors, alpha=0.7,
#                align='center', width=width)
#     axs[0].set_ylabel(f'Distribution [%]')
#     axs[0].set_xticks([])  # clear x tick and tick labels
#
#     # figure setting after plotting
#     plt.tight_layout()
#     fig.subplots_adjust(hspace=0)
#
#     # save figure
#     if output_path is not None:
#         save_fig(fig, output_path, 'shape_mode', build_name, apply_name)
#     return fig, axs

# def scree_plot(latent):
#     """Scree plot for PCA, currently testing use only."""
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#     axs[0].semilogy(latent, 'o', alpha=0.5)
#     axs[1].plot(np.cumsum(latent)/np.sum(latent), 'o', alpha=0.5)
#     return fig, axs
