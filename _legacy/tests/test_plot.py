import pytest

from vampire import util


@pytest.fixture
def properties_df():
    return util.read_pickle('data/extraction/extract_properties_img_set.pickle')


@pytest.fixture
def built_model():
    return util.read_pickle('data/model/Vampire_build.pickle')


@pytest.fixture
def apply_properties_df():
    return util.read_pickle('data/model/Vampire_apply.pickle')


def tst_plot_dendrogram():
    import matplotlib.pyplot as plt
    import numpy as np
    from vampire import extraction, model, plot
    # properties_df = extraction.extract_properties(
    #     r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_LMNA--',
    #     np.array(['tiff']))
    properties_df = extraction.extract_properties(
        r'C:\Files\github-projects\nance-lab-data\microfiber\ogd_severity_segmentations',
        np.array(['cortex', 'otsu']))
    vampire_model = model.Vampire('test-restructure', n_clusters=5, random_state=1)
    vampire_model.fit(properties_df)
    apply_properties_df = vampire_model.transform(properties_df)
    plot.set_plot_style()
    plot.plot_representatives(vampire_model, apply_properties_df, alpha=0.5)
    plt.show()
    fig, axs = plot.plot_distribution_contour_dendrogram(vampire_model,
                                              apply_properties_df,
                                              height_ratio=[4, 1, 1])
    fig.show()
    plot.save_fig(fig, r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_LMNA--',
                  'distribution_contour_dendrogram', '.tif')
    return

def tst_colors():
    import matplotlib.pyplot as plt
    import numpy as np
    colors = [plt.get_cmap('twilight')(cluster_i) for cluster_i in np.linspace(0, 0.95, 20)]
    plt.bar(np.arange(20), np.arange(1, 20+1), color=colors)
    plt.show()
    pass


def tst_build_model():
    import numpy as np
    import pandas as pd
    from vampire import quickstart
    img_info_df = pd.DataFrame({
        'img_set_path': [r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_LMNA--',
                         r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_wildtype'],
        'output_path': [r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_LMNA--',
                         r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_wildtype'],
        'model_name': ['neg', 'wildtype'],
        'n_points': [np.nan, np.nan],
        'n_clusters': [np.nan, np.nan],
        'extension': ['tiff', 'tiff']
    })
    quickstart.fit_models(img_info_df, random_state=1)
    return


def tst_apply_model():
    import pandas as pd
    from vampire import quickstart
    img_info_df = pd.DataFrame({
        'img_set_path': [r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_LMNA--',
                         r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_wildtype'],
        'model_path': [r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_LMNA--\model_neg__tiff.pickle',
                       r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_wildtype\model_wildtype__.pickle'],
        'output_path': [r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_LMNA--',
                         r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_wildtype'],
        'img_set_name': ['neg', 'wildtype'],
        'extension': ['tiff', 'tiff']
    })
    quickstart.transform_datasets(img_info_df)
    return


def tst_time():
    import pandas as pd
    from vampire import quickstart
    import time

    # tic = time.time()
    # img_info_df = pd.DataFrame({
    #     'img_set_path': [r'C:\Files\github-projects\nance-lab-data\microfiber\ogd_severity_segmentations'],
    #     'output_path': [r'C:\Files\github-projects\nance-lab-data\microfiber\restructure-data'],
    #     'model_name': ['vampire-analysis'],
    #     'n_points': [np.nan],
    #     'n_clusters': [np.nan],
    #     'extension': ['tif']
    # })
    # quickstart.build_models(img_info_df, random_state=1)
    # toc = time.time()
    # build_time = toc - tic
    # print(build_time)

    tic = time.time()
    img_info_df = pd.DataFrame({
        'img_set_path': [
            r'C:\Files\github-projects\nance-lab-data\microfiber\ogd_severity_segmentations'],
        'model_path': [
            r'C:\Files\github-projects\nance-lab-data\microfiber\restructure-data\model_vampire-analysis__tif.pickle'],
        'output_path': [
            r'C:\Files\github-projects\nance-lab-data\microfiber\restructure-data'],
        'img_set_name': ['all'],
        'extension': ['tif']
    })
    quickstart.transform_datasets(img_info_df)
    toc = time.time()
    build_time = toc - tic
    print(build_time)


def tst_contour_extraction():
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io, measure
    import cv2

    img_path = r'C:\Files\github-projects\nance-lab-public\vampire_open\Supplementary Data\Example segmented images\MEF_wildtype\xy001c1.tiff'
    img = io.imread(img_path)
    io.imshow(img)
    img = (img == 7282)
    contour_cv = cv2.findContours(img.astype('uint8'),
                               cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)[0][0]
    contour_cv = contour_cv.reshape(-1, 2).T
    # contour_cv = np.flip(contour_cv, axis=1)
    contour_sk = measure.find_contours(img, fully_connected='high')[0].T
    contour_sk = np.flip(contour_sk)

    plt.imshow(img)
    plt.plot(*contour_cv)
    plt.plot(*contour_sk)
    plt.show()
    pass


def tst_region_props():
    import pandas as pd
    from skimage import io, measure
    img = io.imread(r'data/real_img/img1.tif')
    properties = pd.DataFrame(measure.regionprops_table(img, properties=('coords',)))
    pass


def tst_extract_all():
    from vampire import extraction
    extraction.extract_properties(r'C:\Files\github-projects\nance-lab-data\microfiber\ogd_severity_segmentations')


def tst_build_model():
    import numpy as np
    import pandas as pd
    from vampire import quickstart
    img_info_df = pd.DataFrame({
        'img_set_path': [r'C:\Files\github-projects\nance-lab-data\microfiber\ogd_severity_segmentations',
                         r'C:\Files\github-projects\nance-lab-data\microfiber\ogd_severity_segmentations'],
        'output_path': [r'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10',
                        r'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10'],
        'model_name': ['otsu-all', 'yen-all'],
        'n_points': [np.nan, np.nan],
        'n_clusters': [np.nan, np.nan],
        'threshold': ['otsu', 'yen'],
    })
    quickstart.fit_models(img_info_df, random_state=1)
    return


def tst_apply_model():
    import numpy as np
    import pandas as pd
    from vampire import quickstart
    img_info_df = pd.DataFrame({
        'img_set_path': [r'C:\Files\github-projects\nance-lab-data\microfiber\ogd_severity_segmentations',
                         r'C:\Files\github-projects\nance-lab-data\microfiber\ogd_severity_segmentations'],
        'model_path': [r'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10\model_otsu-all__otsu.pickle',
                       r'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10\model_yen-all__yen.pickle'],
        'output_path': [r'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10',
                        r'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10'],
        'img_set_name': ['otsu-all', 'yen-all'],
        'threshold': ['otsu', 'yen'],
    })
    quickstart.transform_datasets(img_info_df)


def tst_process_data():
    import numpy as np
    import pandas as pd
    from vampire import quickstart, util
    import re
    # treatment groups
    control = '4-50-4_|4-50-7_|4-50-10_|4-50-15_'
    ogd30min = '4-56-1_|4-56-2_|4-56-3_|4-56-4_|4-56-5_'
    ogd90min = '4-56-6_|4-56-7_|4-56-8_|4-56-9_|4-56-10_'
    ogd180min = '4-50-1_|4-50-5_|4-50-6_|4-50-12_|4-50-14_'
    ogd90min_azo = '4-59_1_|4-59_2_|4-59_3_|4-59_4_'
    ogd180min_sod = '4-50-2_|4-50-8_|4-50-9_|4-50-11_|4-50-13_'
    treatment_filters = [control, ogd30min, ogd90min, ogd180min, ogd90min_azo, ogd180min_sod]
    treatment_names = ['NC Control', 'OGD 0.5h', 'OGD 1.5h', 'OGD 3.0h', 'OGD 1.5h + AZO', 'OGD 3h + SOD']
    # regions
    region_names = ['cortex', 'thalamus', 'hippocampus']
    # properties
    properties = ['area', 'perimeter', 'circularity', 'aspect_ratio', 'solidity', 'eccentricity', 'extent',
                  'distance_to_centroid']
    otsu_property_df = util.read_pickle(r'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10\otsu\apply-properties_otsu-all_on_otsu-all__otsu.pickle')

    def label_treatment(row):
        for i in range(len(treatment_filters)):
            if re.search(treatment_filters[i], row['filename']):
                return treatment_names[i]

    def label_region(row):
        for i in range(len(region_names)):
            if re.search(region_names[i], row['filename']):
                return region_names[i]

    otsu_property_df['treatment'] = otsu_property_df.transform(label_treatment, axis=1)
    otsu_property_df['region'] = otsu_property_df.transform(label_region, axis=1)

    num_clusters = 5
    # all properties by cluster, all treatment all regions
    for i in range(num_clusters):
        otsu_property_df[(otsu_property_df['cluster_id'] == i)]\
            .drop(['raw_contour', 'normalized_contour'], axis=1)\
            .to_csv(rf'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10\otsu\properties_cluster_{i}_all_treatment_all_regions_all_properties.csv', index=False)

    # 1 property by cluster
    for property in properties:
        property_by_cluster = pd.concat([otsu_property_df[(otsu_property_df['cluster_id'] == i)][property]
                                           for i in range(num_clusters)],
                                          axis=1).reset_index(drop=True)
        property_by_cluster.columns = np.arange(num_clusters)
        property_by_cluster.to_csv(rf'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10\otsu\clusters-vs-{property}_all_treatment_all_regions.csv', index=False)

    # 1 property by cluster, by treatment
    for property in properties:
        for j in range(len(treatment_names)):
            property_by_cluster_treatment = pd.concat([otsu_property_df[(otsu_property_df['cluster_id'] == i)]
                                                       [otsu_property_df[(otsu_property_df['cluster_id'] == i)]['treatment'] == treatment_names[j]]
                                                       [property].reset_index(drop=True)
                                                       for i in range(num_clusters)],
                                                      axis=1)
            property_by_cluster_treatment.columns = np.arange(num_clusters)
            property_by_cluster_treatment.to_csv(
                rf'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10\otsu\clusters-vs-{property}_{treatment_names[j]}_all_regions.csv',
                index=False)

    # 1 property by treatment, by cluster
    for property in properties:
        for i in range(num_clusters):
            property_by_treatment_cluster = pd.concat([otsu_property_df[otsu_property_df['treatment'] == treatment_names[j]]
                                                      [otsu_property_df[otsu_property_df['treatment'] == treatment_names[j]]['cluster_id'] == i]
                                                       [property].reset_index(drop=True)
                                                       for j in range(len(treatment_names))],
                                                      axis=1)
            property_by_treatment_cluster.columns = treatment_names
            property_by_treatment_cluster.to_csv(
                rf'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10\otsu\treatment-vs-{property}_cluster_{i}_all_regions.csv',
                index=False)

    # heatmap, cluster (x) vs treatment (y)
    for property in properties:
        cluster_treatment_pivot = pd.pivot_table(otsu_property_df.groupby(by=['cluster_id', 'treatment']).agg('median'),
                                               values=property,
                                               index=['treatment'],
                                               columns=['cluster_id'])
        cluster_treatment_pivot.to_csv(rf'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10\otsu\cluster-vs-{property}-pivot_all_regions_.csv')

    pass


# def tst_coloring():
#     import numpy as np
#     from vampire import quickstart, util, extraction, coloring
#     filenames = extraction.get_filtered_filenames(r'C:\Files\github-projects\nance-lab-data\microfiber\ogd_severity_segmentations',
#                                                   np.array(['otsu']))
#     img_set = extraction.get_img_set(r'C:\Files\github-projects\nance-lab-data\microfiber\ogd_severity_segmentations', filenames)
#     otsu_property_df = util.read_pickle(
#         r'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10\otsu\apply-properties_otsu-all_on_otsu-all__otsu.pickle')
#     labeled_imgs = coloring.label_imgs(img_set, otsu_property_df)
#     for i, labeled_img in enumerate(labeled_imgs):
#         fig, ax, colors = coloring.color_img(labeled_img)
#         fig.savefig(rf'C:\Files\github-projects\nance-lab-data\microfiber\result-2022-03-10\colored_imgs\colored_{filenames[i]}.png')


def tst_imshow():
    import numpy as np
    from vampire import quickstart, util, extraction, coloring, plot
    import matplotlib.pyplot as plt
    from skimage import io
    img = io.imread(r'C:\Files\github-projects\nance-lab-data\rat-microglia\raw\4-50-11_40x_cortex_1.tif')
    plot.set_plot_style()
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    plt.axis('off')
    plt.imshow(img[:, :, 0])
    plt.show()
    plt.axis('off')
    plt.imshow(img[:, :, 1])
    plt.show()
    pass


def tst_extract_all_ferret():
    from vampire import extraction
    extraction.extract_properties(r'C:\Files\github-projects\nance-lab-data\ferret-microglia\segmented')


def tst_build_model_ferret():
    import numpy as np
    import pandas as pd
    from vampire import quickstart
    img_info_df = pd.DataFrame({
        'img_set_path': [r'C:\Files\github-projects\nance-lab-data\ferret-microglia\segmented'],
        'output_path': [r'C:\Files\github-projects\nance-lab-data\ferret-microglia\vampire-crosscheck\output-new'],
        'model_name': ['ferret'],
        'n_points': [np.nan],
        'n_clusters': [np.nan],
    })
    quickstart.fit_models(img_info_df, random_state=1)
    return


def tst_apply_model_ferret():
    import numpy as np
    import pandas as pd
    from vampire import quickstart
    img_info_df = pd.DataFrame({
        'img_set_path': [r'C:\Files\github-projects\nance-lab-data\ferret-microglia\segmented'],
        'model_path': [r'C:\Files\github-projects\nance-lab-data\ferret-microglia\vampire-crosscheck\output-new\model_ferret__.pickle'],
        'output_path': [r'C:\Files\github-projects\nance-lab-data\ferret-microglia\vampire-crosscheck\output-new'],
        'img_set_name': ['ferret'],
    })
    quickstart.transform_datasets(img_info_df)


def tst_model():
    import vampire as vp
    model = vp.util.read_pickle(r'C:\Files\github-projects\nance-lab-data\ferret-microglia\vampire-crosscheck\output-new\model_ferret__.pickle')
    pass