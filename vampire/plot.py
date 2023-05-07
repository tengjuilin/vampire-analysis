import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy import cluster

from . import analysis


def set_plot_style():
    """
    Set matplotlib plot settings of ``rcParams`` for better visualization.

    References
    ----------
    [1] Customizing Matplotlib with style sheets and rcParams.
    https://matplotlib.org/stable/tutorials/introductory/customizing.html

    """
    plt.rcParams.update({
        'font.family': 'Arial',
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


def save_fig(
    fig,
    output_path,
    fig_type,
    extension='.png',
    model_name=None,
    apply_name=None
):
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
    extension : str, optional
        File extension.
    model_name : str, optional
        Name of the built model.
    apply_name : str, optional
        Name of the image set being applied to.

    """
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if model_name is None:
        fig_path = os.path.join(output_path, f'{fig_type}_{time_stamp}{extension}')
    else:
        if apply_name is None:  # build model
            fig_path = os.path.join(output_path, f'{fig_type}_build_{model_name}{extension}')
        else:  # apply model
            fig_path = os.path.join(output_path, f'{fig_type}_apply_{model_name}_on_{apply_name}{extension}')

    if os.path.exists(fig_path):
        fig_name, extension = os.path.splitext(fig_path)
        fig.savefig(f'{fig_name}_{time_stamp}{extension}', dpi=300)
    else:
        fig.savefig(fig_path, dpi=300)
    return


def plot_dendrogram(model, ax=None, fig_size=(6, 2)):
    """
    Plots dendrogram of mean contours.

    Parameters
    ----------
    model : vampire.model.Vampire
        Built VAMPIRE model.
    ax : Axes, optional
        Figure axis to be plotted on. Cannot be not None with ``output_path``
        at the same time.
    fig_size : (float, float), optional
        Width, height in inches. Default (6, 2).

    Returns
    -------
    ax : matplotlib.axes.Axes

    See Also
    --------
    scipy.cluster.hierarchy.dendrogram

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    cluster.hierarchy.set_link_color_palette(['k'])
    cluster.hierarchy.dendrogram(
        model.linkage_matrix,
        ax=ax,
        p=0,
        truncate_mode='lastp',
        orientation='bottom',
        above_threshold_color='k'
    )
    ax.axis('off')
    return ax


def plot_contours(
    model,
    apply_properties_df=None,
    contour_scale=3,
    ax=None,
    fig_size=(6, 2),
    colors=None,
    alpha=1,
    lw=2
):
    """
    Plots mean contours.

    Parameters
    ----------
    model : vampire.model.Vampire
        Built VAMPIRE model.
    apply_properties_df : DataFrame, optional
        Properties output of VAMPIRE model applied to data.
    contour_scale : float, optional
        Scale of the contour shapes.
    ax : Axes, optional
        Figure axis to be plotted on. Cannot be not None with ``output_path``
        at the same time.
    fig_size : (float, float), optional
        Width, height in inches. Default (6, 2).
    colors : str or list, optional
        Colors of mean contours.
    alpha : float, optional
        Alpha of mean contours.
    lw : float, optional
        Line width of mean contours.

    Returns
    -------
    ax : matplotlib.axes.Axes

    """
    if apply_properties_df is None:
        mean_cluster_contours = model.mean_cluster_contours
    else:
        contours = np.vstack(apply_properties_df['normalized_contour'].to_numpy())
        cluster_id_df = apply_properties_df[['cluster_id', 'distance_to_centroid']]
        labeled_contours_df = analysis.get_labeled_contours_df(contours, cluster_id_df)
        mean_cluster_contours = analysis.get_mean_cluster_contours(labeled_contours_df)
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    if colors is None:
        colors = [plt.get_cmap('twilight')(cluster_i)
                  for cluster_i in np.linspace(0.1, 0.9, model.n_clusters)]
    elif type(colors) == str:
        colors = [colors] * model.n_clusters

    x_first = 5  # offset of first contour
    x_offset = 10  # offset between contours

    for i in range(model.n_clusters):
        # read in contour coordinates
        x = mean_cluster_contours[i, :model.n_points]
        y = mean_cluster_contours[i, model.n_points:]
        # form close shape when plotting
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        # place shape into right location
        x = x * contour_scale + x_first + x_offset * i
        y = y * contour_scale
        # plot shape of objects corresponding to the branches
        ax.plot(x, y, color=colors[i], alpha=alpha, lw=lw)
    ax.axis('equal')
    ax.axis('off')
    return ax


def plot_representatives(
    model, apply_properties_df,
    n_samples=10,
    random_state=None,
    ax=None,
    fig_size=(17, 2),
    colors=None,
    alpha=None,
    lw=None
):
    """
    Plots representative object contours.

    Parameters
    ----------
    model : vampire.model.Vampire
        Built VAMPIRE model.
    apply_properties_df : DataFrame, optional
        Properties output of VAMPIRE model applied to data.
    n_samples : int, optional
        Number of sample drawn from each cluster. Default 10. If n_samples >
        number of total available samples in the smallest cluster, it is
        set to the that number.
    random_state : int, optional
        Random state of sampling representative contours.
    ax : Axes, optional
        Figure axis to be plotted on. Cannot be not None with ``output_path``
        at the same time.
    fig_size : (float, float), optional
        Width, height in inches. Default (17, 2).
    colors : str or list, optional
        Colors of representative contours of clusters.
    alpha : float, optional
        Alpha of representative contours.
    lw : float, optional
        Line width of representative contours.

    Returns
    -------
    ax : matplotlib.axes.Axes

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    if colors is None:
        colors = [plt.get_cmap('twilight')(cluster_i)
                  for cluster_i in np.linspace(0.1, 0.9, model.n_clusters)]
    elif type(colors) == str:
        colors = [colors] * model.n_clusters

    x_offset = 5  # move center of another cluster to new location

    # determine sample size
    cluster_id = apply_properties_df['cluster_id'].values
    unique, counts = np.unique(cluster_id, return_counts=True)
    max_n_samples = np.min(counts)
    if n_samples > max_n_samples:
        n_samples = max_n_samples

    # sample contours from all clusters
    contours = np.vstack(apply_properties_df['normalized_contour'].to_numpy())
    cluster_id_df = apply_properties_df['cluster_id']
    labeled_contours_df = analysis.get_labeled_contours_df(contours, cluster_id_df)
    all_cluster_samples_df = labeled_contours_df.groupby('cluster_id') \
        .sample(
        n=n_samples,
        random_state=random_state
    )

    # plotting each sample contour
    for cluster_i in range(model.n_clusters):
        # sample contour in current cluster
        cluster_samples_df = all_cluster_samples_df[all_cluster_samples_df['cluster_id'] == cluster_i]
        cluster_samples = cluster_samples_df.drop(columns='cluster_id').values
        for sample_i in range(n_samples):
            # plot each sample contour by...
            # read in sample contour coordinates
            x = cluster_samples[sample_i, :model.n_points]
            y = cluster_samples[sample_i, model.n_points:]
            # form close shape when plotting
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            # place shape into right location
            x = x + x_offset * cluster_i
            y = y
            # sample shape of objects corresponding to the clusters
            ax.plot(x, y, color=colors[cluster_i], alpha=alpha, lw=lw)
    plt.axis('equal')
    plt.axis('off')
    return ax


def plot_distribution(properties_df, ax=None):
    """
    Plots the distribution of mean contours in a bar graph.

    Parameters
    ----------
    properties_df : DataFrame
        DataFrame containing contour coordinates of objects and a column
        named 'cluster_id' that indicates the cluster that an object belongs.
    ax : Axes, optional
        Figure axis to be plotted on. Cannot be not None with ``output_path``
        at the same time.

    Returns
    -------
    ax : matplotlib.axes.Axes

    """
    if ax is None:
        fig, ax = plt.subplots()

    distribution = analysis.get_distribution(properties_df) * 100  # unit: percent
    n_clusters = len(distribution)

    x_first = 5  # offset of first contour
    x_offset = 10  # offset between contours
    width = x_offset / 2
    x = np.arange(
        x_first,
        n_clusters * x_offset + x_offset / 2,
        x_offset
    )
    colors = [
        plt.get_cmap('twilight')(cluster_i)
        for cluster_i in np.linspace(0.1, 0.9, n_clusters)
    ]
    ax.bar(
        x=x,
        height=distribution,
        color=colors,
        align='center',
        width=width
    )
    ax.set_ylabel(f'Distribution [%]')
    ax.set_xticks([])  # clear x tick and tick labels

    # figure settings after plotting
    plt.tight_layout()
    return ax


def plot_contour_dendrogram(model, fig_size=(6, 2)):
    """
    Plots dendrogram with mean contours.

    Parameters
    ----------
    model : vampire.model.Vampire
        Built VAMPIRE model.
    fig_size : (float, float), optional
        Width, height in inches. Default (6, 2).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : matplotlib.axes.Axes

    See Also
    --------
    plot_dendrogram, plot_contours

    """
    fig, axs = plt.subplots(2, 1, figsize=fig_size, frameon=False, sharex='all')
    plot_dendrogram(model, ax=axs[1])
    plot_contours(model, ax=axs[0])
    return fig, axs


def plot_distribution_contour(
    model,
    apply_properties_df=None,
    fig_size=(5, 5),
    height_ratio=(4, 1)
):
    """
    Plots the distribution of mean contours in a bar graph with labeling
    of mean contours.

    Parameters
    ----------
    model : vampire.model.Vampire
        Built VAMPIRE model.
    apply_properties_df : DataFrame, optional
        Properties output of VAMPIRE model applied to data.
    fig_size : (float, float), optional
        Width, height in inches. Default (5, 5).
    height_ratio : list[float], optional
        Ratio between height of distribution plot, shape mode contours, and
        shape mode dendrogram. Default [4, 1, 1]. Recommended values:

            * [4, 1, 1] for 5 clusters (shape modes)
            * [4, 0.5, 1] for 10 clusters (shape modes)
            * [4, 0.35, 1] for 15 clusters (shape modes)

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : matplotlib.axes.Axes

    See Also
    --------
    plot_contours, plot_distribution

    """
    fig, axs = plt.subplots(
        2, 1,
        figsize=fig_size,
        sharex='all',
        gridspec_kw={'height_ratios': height_ratio}
    )
    plot_contours(model, ax=axs[1])
    if apply_properties_df is None:
        plot_distribution(model.cluster_id_df, ax=axs[0])
    else:
        plot_distribution(apply_properties_df, ax=axs[0])
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, axs


def plot_distribution_contour_dendrogram(
    model,
    apply_properties_df=None,
    fig_size=(5, 5),
    height_ratio=(4, 1, 1)
):
    """
    Plots the distribution of mean contours in a bar graph with labeling
    of mean contours and their dendrogram.

    Parameters
    ----------
    model : vampire.model.Vampire
        Built VAMPIRE model.
    apply_properties_df : DataFrame, optional
        Properties output of VAMPIRE model applied to data.
    fig_size : (float, float), optional
        Width, height in inches. Default (6, 2).
    height_ratio : list[float], optional
        Ratio between height of distribution plot, shape mode contours, and
        shape mode dendrogram. Default [4, 1, 1]. Recommended values:

            * [4, 1, 1] for 5 clusters (shape modes)
            * [4, 0.5, 1] for 10 clusters (shape modes)
            * [4, 0.35, 1] for 15 clusters (shape modes)

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : matplotlib.axes.Axes

    See Also
    --------
    plot_dendrogram, plot_contours, plot_distribution

    """
    # figure structure
    height_ratio = list(height_ratio)  # prevent mutable default argument
    fig, axs = plt.subplots(
        3, 1,
        figsize=fig_size,
        sharex='all',
        gridspec_kw={'height_ratios': height_ratio}
    )

    if apply_properties_df is None:  # build model plot
        plot_dendrogram(model, ax=axs[2])
        plot_contours(model, ax=axs[1])
        plot_distribution(model.cluster_id_df, ax=axs[0])
    else:  # apply model plot
        plot_dendrogram(model, ax=axs[2])
        plot_contours(model, ax=axs[1])
        plot_contours(
            model,
            apply_properties_df, ax=axs[1],
            colors='black',
            alpha=0.3
        )
        plot_distribution(apply_properties_df, ax=axs[0])

    # figure settings after plotting
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, axs
