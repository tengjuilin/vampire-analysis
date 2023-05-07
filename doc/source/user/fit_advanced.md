(fit_advanced)=

# Build Models: Advanced

In this section, we discuss the flexibility of input format, the
defaults of required information, and the use of filter information
for building models.

(_build_advanced_input_format)=

## Input format

We can store image set information using xlsx, csv, or
DataFrame.

### xlsx input

In the :ref:`build model basics <build_basics_building_models>` section, we used an
xlsx excel file as the carrier of the image set information:

| img_set_path             | output_path        | model_name        | num_points | num_clusters | channel |
|-|-|-|-|-|-|
| `C:\vampire-ogd\both`  | `C:\vampire-ogd` | control-ogd-30min | 50         | 5            | c1      |

We converted the xlsx file to a DataFrame that `vampire.model.build_model()` can take in:

```python
>>> import pandas as pd
>>> import vampire as vp
```

```python
>>> build_df = pd.read_excel(r'C:\vampire-ogd\build.xlsx')
>>> vp.model.fit_models(build_df, random_state=1)
```

### csv input

`vampire-analysis` is also compatible with csv file to
store the information if it matches the workflow:

```csv
# build.csv
img_set_path, output_path, model_name, num_points, num_clusters, channel
C:\vampire-ogd\both, C:\vampire-ogd, control-ogd-30min, 50, 5, c1
```

We can then convert the csv file to a DataFrame that `vampire.model.build_model` can take in:

```python
>>> build_df = pd.read_csv(r'C:\vampire-ogd\build.csv')
>>> vp.model.fit_models(build_df, random_state=1)
```

### DataFrame input

For pipelines fully automated by Python, direct use of DataFrame
is encouraged:

```python
>>> d = {'img_set_path': [r'C:\vampire-ogd\both'],
...      'output_path': [r'C:\vampire-ogd']
...      'model_name': ['control-ogd-30min'],
...      'num_points': [50],
...      'num_clusters': [5],
...      'channel': ['c1']}
>>> build_df = pd.DataFrame(data=d)
>>> vp.model.fit_models(build_df, random_state=1)
...      'model_name': ['control-ogd-30min'],
...      'num_points': [50],
...      'num_clusters': [5],
...      'channel': ['c1']}
>>> build_df = pd.DataFrame(data=d)
>>> vp.model.fit_models(build_df, random_state=1)
...      'model_name': ['control-ogd-30min'],
...      'num_points': [50],
...      'num_clusters': [5],
...      'channel': ['c1']}
>>> build_df = pd.DataFrame(data=d)
>>> vp.model.build_models(build_df, random_state=1)
```

## Input file structure

The input file for building models consists of required information in the
first 5 columns and optional filter information in additional columns,
if needed.

```{seealso}

    :func:`vampire.model.build_models`

.. _build_advanced_required_info:
```

### Defaults of required information

Here, we discuss rules of the required information and their default values
and provide some examples.

#### Rules

The input DataFrame `img_info_df` must contain, *in order*, the 5
required columns of

- `img_set_path` : str
  - Path to the image set(s) to be used to build model.
- `output_path` : str, default
  - Path of the directory used to output model and figures. Defaults to the path of the directory of each image set.
- `model_name` : str, default
  - Name of the model. Defaults to time of function call.
- `num_points` : int, default
  - Number of sample points of object contour. Defaults to 50.
- `num_clusters` : int, default
  - Number of clusters of K-means clustering. Defaults to 5. Recommended range [2, 10].

in the first 5 columns. The default values are used in default columns when

- the space is left blank in csv or xlsx file before
    converting to DataFrame
- the space is `None`/`np.NaN` in the DataFrame

#### Example: default values

For example, the csv or xlsx file with content

```{tip}
| img_set_path            | output_path | model_name | num_points | num_clusters | channel |
|-|-|-|-|-|-|
| `C:\vampire-ogd\both` |             |            |            |              | c1      |
```

is equivalent to

```{tip}
| img_set_path             | output_path             | model_name          | num_points | num_clusters | channel |
|-|-|-|-|-|-|
| `C:\vampire-ogd\both`  | `C:\vampire-ogd\both` | 2021-08-04_13-45-37 | 50         | 5            | c1      |
```

because

- the default of `output_path` is `img_set_path`
- the default of `model_name` is the model build time, which is,
  for example, 2021-08-04_13-45-37
- the default of `num_points` is 50
- the default of `num_clusters` is 5

#### Example: order matters

**The five required columns must appear in order.**

For example, if we want to input the following information

```{tip}
| img_set_path            | output_path        | model_name | num_points | num_clusters | channel |
|-|-|-|-|-|-|
| `C:\vampire-ogd\both` | `C:\vampire-ogd` | ogd-both   |            |              | c1      |
```

do not shuffle the columns

```{error}
| model_name | img_set_path            | num_clusters | output_path        | num_points | channel |
|-|-|-|-|-|-|
| ogd-both   | `C:\vampire-ogd\both` |              | `C:\vampire-ogd` |            | c1      |
```

it will NOT give the desired output, because `vampire-analysis` will read
the table in ordered sequence as:

```{error}
| img_set_path | output_path             | model_name | num_points         | num_clusters | channel |
|-|-|-|-|-|-|
| ogd-both     | `C:\vampire-ogd\both` |            | `C:\vampire-ogd` |              | c1      |
```

which makes no sense.

#### Example: column headings and default values

**Even when you have left the columns blank for default, the
column heading has to appear as a placeholder.**

For example, the table without required default column headings

```{error}
| img_set_path            | channel |
|-|-|
| `C:\vampire-ogd\both` | c1      |
```

will throw `ValueError: Input DataFrame does not have enough number
of columns.` Instead, use column headings as placeholders:

```{tip}
| img_set_path            | output_path | model_name | num_points | num_clusters | channel |
|-|-|-|-|-|-|
| `C:\vampire-ogd\both` |             |            |            |              | c1      |
```

#### Example: multiple image sets and defaults

You may specify multiple image sets used to build model with flexible
use of defaults:

```{tip}
| img_set_path            | output_path        | model_name | num_points | num_clusters | channel |
|-|-|-|-|-|-|
| `C:\vampire-ogd\both` |                    | ogd-both   | 40         |              | c1      |
| `C:\vampire-ogd\both` | `C:\vampire-ogd` |            | 80         | 10           | c1      |
| `C:\vampire-ogd\both` |                    |            |            |              | c1      |
| `C:\vampire-ogd\both` |                    |  seven     |            | 7            | c1      |
```

which is equivalent to

```{tip}
| img_set_path             | output_path             | model_name          | num_points | num_clusters | channel |
|-|-|-|-|-|-|
| `C:\vampire-ogd\both`  | `C:\vampire-ogd\both` | ogd-both            | 40         | 5            | c1      |
| `C:\vampire-ogd\both`  | `C:\vampire-ogd`      | 2021-08-04_13-45-37 | 80         | 10           | c1      |
| `C:\vampire-ogd\both`  | `C:\vampire-ogd\both` | 2021-08-04_13-46-11 | 50         | 5            | c1      |
| `C:\vampire-ogd\both`  | `C:\vampire-ogd\both` |  seven              | 50         | 7            | c1      |
```

Note that because the analysis takes some time, the model name that defaults
to the build model time will differ for different image sets.

(_build_advanced_filter_info)=

### Use of filter information

Here, we discuss rules and example use of filter information in the
optional columns.

#### Rules

The input DataFrame `img_info_df` could contain any number (none
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

#### Example: no filter

Suppose we have images

- `4-50-1_40x_cortex_1_c1.tiff`
- `4-50-1_40x_hippocampus_1_c1.png`
- `4-50-1_40x_hippocampus_1_c2.jpeg`

We want to analyze all images in the image set folder.
We can simply not have any columns at the right of the required columns
to signify we are not using any filters. That is, all images, with supported
extensions `'.tiff', '.tif', '.jpeg', '.jpg', '.png', '.bmp', '.gif'`,
will be used in building the model:

```{tip}
| img_set_path            | output_path | model_name | num_points | num_clusters |
|-|-|-|-|-|
| `C:\vampire-ogd\both` |             |            |            |              |
```

All the files are used to build model:

- `4-50-1_40x_cortex_1_c1.tiff`
- `4-50-1_40x_hippocampus_1_c1.png`
- `4-50-1_40x_hippocampus_1_c2.jpeg`

#### Example: one filter

Suppose we have images

- `4-50-1_40x_cortex_1_c1.tiff`
- `4-50-1_40x_cortex_1_c2.tiff`
- `4-50-1_40x_cortex_1_c3.tiff`

and we only want to include channel 1 images, which contain `c1` in their
filenames, we can use an optional column as filter:

```{tip}
| img_set_path            | output_path | model_name | num_points | num_clusters | channel |
|-|-|-|-|-|-|
| `C:\vampire-ogd\both` |             |            |            |              | c1      |
```

so that only channel 1 image is used to build model:

- `4-50-1_40x_cortex_1_c1.tiff`

#### Example: multiple filters

Suppose we have images

- `4-50-1_40x_cortex_1_c1.tiff`
- `4-50-1_40x_cortex_1_c2.tiff`
- `4-50-1_40x_cortex_1_c3.tiff`
- `4-50-1_40x_hippocampus_1_c1.tiff`
- `4-50-1_40x_hippocampus_1_c2.tiff`
- `4-50-1_40x_hippocampus_1_c3.tiff`

and we want to include images that are in channel 1 AND in hippocampus,
which contain `c1` and `hippocampus` in their
filenames, we can use an optional columns as an AND filter:

```{tip}
| img_set_path            | output_path | model_name | num_points | num_clusters | channel | region      |
|-|-|-|-|-|-|-|
| `C:\vampire-ogd\both` |             |            |            |              | c1      | hippocampus |
```

so that image whose filename contains `c1` and `hippocampus` is used:

- `4-50-1_40x_hippocampus_1_c1.tiff`

```{note}
The headings of the optional columns do not affect the analysis.
Use headings that are descriptive for your purposes.
```

```{warning}
The optional columns serve as an AND filter, which means only images
that satisfy condition 1 AND condition 2 will be used. To illustrate
this, see the next example.
```

#### Example: AND filter

Suppose we have images

- `4-50-1_40x_cortex_1_c1.tiff`
- `4-50-1_40x_cortex_1_c2.tif`
- `4-50-1_40x_cortex_1_c3.tiff`
- `4-50-1_40x_hippocampus_1_c1.png`
- `4-50-1_40x_hippocampus_1_c2.png`
- `4-50-1_40x_hippocampus_1_c3.jpeg`

If the image set contains images with different file extensions, and we
only want a particular file extension, say `tiff`, to be used in
building the model, we can use

```{tip}
| img_set_path            | output_path | model_name | num_points | num_clusters | extension |
|-|-|-|-|-|-|
| `C:\vampire-ogd\both` |             |            |            |              | tiff      |
```

so that only images whose filename contains `tiff` are used:

- `4-50-1_40x_cortex_1_c1.tiff`
- `4-50-1_40x_cortex_1_c3.tiff`

However, we cannot use the optional columns to filter multiple extensions,
such as

```{error}
| img_set_path            | output_path | model_name | num_points | num_clusters | extension1 | extension2 | extension3 |
|-|-|-|-|-|-|-|-|
| `C:\vampire-ogd\both` |             |            |            |              | tiff       | tif        | png        |
```

because what we wanted is files with extension `tiff` OR `tif` OR `png`,
but `vampire-analysis` is looking for files that contains `tiff` AND
`tif` AND `png`. None of the image satisfied such condition.
OR filtering is currently not supported.

#### Example: filter combinations

Suppose we have images

- `4-50-1_40x_cortex_1_c1.tiff`
- `4-50-1_40x_cortex_1_c2.tiff`
- `4-50-1_40x_cortex_2_c1.tiff`
- `4-50-1_40x_cortex_2_c2.tiff`
- `4-50-1_40x_hippocampus_1_c1.tiff`
- `4-50-1_40x_hippocampus_1_c2.tiff`
- `4-50-1_40x_hippocampus_2_c1.tiff`
- `4-50-1_40x_hippocampus_2_c2.tiff`

We want to build models from this image set using a combination of
channels and regions, as well as the image set as a whole. We
can accomplish this with:

```{tip}

| img_set_path            | output_path | model_name | num_points | num_clusters | channel | region      |
|-|-|-|-|-|-|-|
| `C:\vampire-ogd\both` |             |            |            |              | c1      | cortex      |
| `C:\vampire-ogd\both` |             |            |            |              | c2      | hippocampus |
| `C:\vampire-ogd\both` |             |            |            |              | c1      |             |
| `C:\vampire-ogd\both` |             |            |            |              |         | hippocampus |
| `C:\vampire-ogd\both` |             |            |            |              |         |             |
```

so that the 1st model is based on

- `4-50-1_40x_cortex_1_c1.tiff`
- `4-50-1_40x_cortex_2_c1.tiff`

the 2nd model is based on

- `4-50-1_40x_hippocampus_1_c2.tiff`
- `4-50-1_40x_hippocampus_2_c2.tiff`

the 3rd model is based on

- `4-50-1_40x_cortex_1_c1.tiff`
- `4-50-1_40x_cortex_2_c1.tiff`
- `4-50-1_40x_hippocampus_1_c1.tiff`
- `4-50-1_40x_hippocampus_2_c1.tiff`

the 4th model is based on

- `4-50-1_40x_hippocampus_1_c1.tiff`
- `4-50-1_40x_hippocampus_1_c2.tiff`
- `4-50-1_40x_hippocampus_2_c1.tiff`
- `4-50-1_40x_hippocampus_2_c2.tiff`

the 5th model is based on the whole image set

- `4-50-1_40x_cortex_1_c1.tiff`
- `4-50-1_40x_cortex_1_c2.tiff`
- `4-50-1_40x_cortex_2_c1.tiff`
- `4-50-1_40x_cortex_2_c2.tiff`
- `4-50-1_40x_hippocampus_1_c1.tiff`
- `4-50-1_40x_hippocampus_1_c2.tiff`
- `4-50-1_40x_hippocampus_2_c1.tiff`
- `4-50-1_40x_hippocampus_2_c2.tiff`

## Conclusion

We have explored options to provide input information to build models using csv, xlsx, and DataFrame. We also looked at the requirements and examples of required and optional filtering information for building models.

Next, we will look at some advanced options when specifying image set information for applying models.
