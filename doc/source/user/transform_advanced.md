(transform_advanced)=

# Apply Models: Advanced

In this section, we discuss the flexibility of input format, the
defaults of required information, and the use of filter information
for applying models.

## Input format

Like building models, applying models is also compatible with  ``.xlsx``,
``.csv``, or ``DataFrame`` image set information input. Please refer to
:ref:`build models input format <build_advanced_input_format>`, and use
``vampire.model.apply_models()``

## Input file structure

The input file for applying models consists of required information in the
first 4 columns and optional filter information in additional columns,
if needed.

.. seealso::

    :func:`vampire.model.apply_models`

### Defaults of required information

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
The default values are used in default columns when

    - the space is left blank in ``.csv`` or ``.xlsx`` file before
      converting to ``DataFrame``
    - the space is ``None``/``np.NaN`` in the ``DataFrame``

For examples for defaults, please refer to
:ref:`build models required information <build_advanced_required_info>`.

### Use of filter information

Like building models, applying models has the same guidelines for filter
information. Please refer to
:ref:`build models filter information <build_advanced_filter_info>`.