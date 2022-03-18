import matplotlib.pyplot as plt
import numpy as np


def label_imgs(img_set, properties_df):
    """
    Label objects in the images within the set according to clusters.

    Parameters
    ----------
    img_set : list[ndarray]
        Set of images to be labeled.
    properties_df : DataFrame
        Properties of objects from ``img_set``.
        Contains labels and cluster_ids.

    Returns
    -------
    labeled_imgs : list[ndarray]
        Image set with objects labeled according to clusters.

    """
    img_ids = np.unique(properties_df['image_id'])
    labeled_imgs = []
    for i, img_id in enumerate(img_ids):
        img_df = properties_df[properties_df['image_id'] == img_id]
        labeled_img = label_img(img_set[i], img_df)
        labeled_imgs.append(labeled_img)
    return labeled_imgs


def label_img(img, img_df):
    """
    Label objects in the image according to clusters.

    Parameters
    ----------
    img : ndarray
        Image to be labeled.
    img_df : DataFrame
        Properties of objects from ``img``.
        Contains labels and cluster_ids.

    Returns
    -------
    labeled_img : ndarray
        Image with objects labeled according to clusters.

    """
    AVOID_OVERRIDE_NUM = 2**16
    masks = []
    cluster_ids = np.unique(img_df['cluster_id'])
    for cluster_id in cluster_ids:
        cluster_df = img_df[img_df['cluster_id'] == cluster_id]
        mask = np.isin(img, cluster_df['label'])
        # 0 reserved for background, switch to 1 indexing
        mask = mask * AVOID_OVERRIDE_NUM * (cluster_id + 1)
        masks.append(mask)
    labeled_img = sum(masks) / AVOID_OVERRIDE_NUM
    return labeled_img


def color_img(img, background=0, cmap=None, background_color=None):
    """
    Plot cluster-labeled images.

    Parameters
    ----------
    img : ndarray
        Cluster-labeled image.
    background : int, optional
        Background value. Default 0.
    cmap : str, optional
        Matplotlib colormap name.
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    background_color : str, optional
        Color name for background.

    Returns
    -------
    fig : matplotlib.pyplot.figure
    ax : matplotlib.axes.Axes
    colors : ndarray
        Colors used to color each cluster

    """
    if cmap is None:
        cmap = plt.get_cmap('twilight').copy()
    else:
        cmap = plt.get_cmap(cmap).copy()
    if background_color is None:
        cmap.set_bad(color='white')
    else:
        cmap.set_bad(color=background_color)

    # avoid modifying img in outer scope with inplace operations
    img = np.copy(img)

    # assign each cluster a label that's normalized for cmap
    cluster_ids = np.unique(img)
    cluster_ids = np.delete(cluster_ids,
                            np.where(cluster_ids == background))
    num_clusters = len(cluster_ids)
    replaced_labels = np.linspace(0.1, 0.9, num_clusters)
    for i, replaced_label in enumerate(replaced_labels):
        img[img == cluster_ids[i]] = replaced_label

    # make background "bad" so it displays background_color
    img[img == background] = np.nan

    # plot cluster-labeled img
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap=cmap, vmax=1, vmin=0)
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False,
                   top=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False)
    plt.tight_layout(pad=0)

    # colors used for labeling
    colors = cmap(replaced_labels)

    return fig, ax, colors
