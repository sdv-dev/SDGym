"""NVIDIA NeMo Data Designer integration."""

import logging
import sys

import numpy as np
import pandas as pd
from sdv.metadata import Metadata

from sdgym.synthesizers.base import BaselineSynthesizer

LOGGER = logging.getLogger(__name__)

PII_PLACEHOLDER_PREFIX = 'sdgym-pii-'
MIN_PYTHON_VERSION = (3, 10)
DEFAULT_MODEL_ALIAS = 'text'
DEFAULT_MODEL_ID = 'nvidia/nvidia-nemotron-nano-9b-v2'
DEFAULT_MODEL_PROVIDER = 'nvidia'
MAX_TEXT_EXAMPLES = 3
MAX_TEXT_EXAMPLE_LENGTH = 100


MISSING_PROPORTION_ATTRIBUTES = (
    'missing_values_proportion',
    'missing_proportion',
    'null_proportion',
)
# sdtypes that can be useful as context
TEXT_CONTEXT_SDTYPES = {'numerical', 'categorical', 'boolean', 'datetime'}


def _import_data_designer():
    if sys.version_info < MIN_PYTHON_VERSION:
        raise ImportError('DataDesigner only supports python >= 3.10')

    try:
        import data_designer.config as data_designer_config
    except ImportError as exception:
        raise ImportError(
            "To use 'DataDesignerSynthesizer' you have to install the extra "
            "dependencies by running pip install sdgym['data_designer']"
        ) from exception

    return data_designer_config


def _get_table_metadata(metadata, table_name=None):
    """Return the metadata of ``table_name`` from a ``Metadata`` object."""
    if not isinstance(metadata, Metadata):
        raise TypeError(
            f'Expected sdv.Metadata, got {type(metadata).__name__}. '
            'DataDesignerSynthesizer only supports single table metadata.'
        )

    if table_name is None:
        if len(metadata.tables) != 1:
            raise ValueError(
                f'Metadata has {len(metadata.tables)} tables, please provide a `table_name`. '
                'DataDesignerSynthesizer only supports single table metadata.'
            )

        table_name = metadata._get_single_table_name()

    if table_name not in metadata.tables:
        raise ValueError(f"Table '{table_name}' is not present in the metadata.")

    return metadata.tables[table_name]


def _get_missing_proportion(column_data, column_metadata):
    if column_metadata.get('range_is_nullable') is False or len(column_data) == 0:
        return 0.0

    return float(column_data.isna().mean())


def _attach_missing_proportion(column_config, missing_proportion):
    if not missing_proportion:
        return

    for attribute in MISSING_PROPORTION_ATTRIBUTES:
        if hasattr(column_config, attribute):
            setattr(column_config, attribute, missing_proportion)
            return

    LOGGER.debug(
        f"Column '{column_config.name}' has {missing_proportion:.1%} missing values but the "
        'DataDesignerSynthesizer has no missing value setting.'
    )


def _create_default_model_config(dd, model_alias):
    return dd.ModelConfig(
        alias=model_alias,
        model=DEFAULT_MODEL_ID,
        provider=DEFAULT_MODEL_PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(temperature=0.85, top_p=0.95),
    )


def _create_categorical_config(dd, column_name, column_data, column_metadata):
    """Map a categorical column to a weighted ``category`` sampler."""
    value_counts = column_data.dropna().value_counts()
    values = column_metadata.get('range_values')
    if values is None:
        values = list(value_counts.index)

    if not values:
        LOGGER.warning(
            f"Column '{column_name}' has no observed values, using a placeholder sampler."
        )
        return _create_placeholder_config(dd, column_name)

    values = [value.item() if isinstance(value, np.generic) else value for value in values]
    weights = [float(value_counts.get(value, 0)) for value in values]
    params = dd.CategorySamplerParams(values=values, weights=weights if sum(weights) else None)
    return dd.SamplerColumnConfig(
        name=column_name, sampler_type=dd.SamplerType.CATEGORY, params=params
    )


def _create_boolean_config(dd, column_name, column_data):
    """Map a boolean column to a ``bernoulli`` sampler."""
    values = column_data.dropna()
    probability = float(values.astype(bool).mean()) if not values.empty else 0.5
    params = dd.BernoulliSamplerParams(p=probability)
    return dd.SamplerColumnConfig(
        name=column_name, sampler_type=dd.SamplerType.BERNOULLI, params=params
    )


def _get_decimal_places(values):
    """Return the largest number of decimal places observed in ``values``."""
    decimals = 0
    for value in values.head(1000):
        _, _, fraction = repr(float(value)).partition('.')
        if fraction and fraction != '0':
            decimals = max(decimals, len(fraction))

    return decimals


def _create_numerical_config(dd, column_name, column_data, column_metadata):
    """Map a numerical column to a ``uniform`` sampler.

    The range and precision come from the metadata ``range_min``, ``range_max`` and
    ``decimal_places`` when present, otherwise they are computed from the data.
    """
    values = pd.to_numeric(column_data, errors='coerce').dropna()
    if values.empty:
        LOGGER.warning(f"Column '{column_name}' has no numeric values, sampling zeros.")
        values = pd.Series([0.0])

    low = column_metadata.get('range_min', values.min())
    high = column_metadata.get('range_max', values.max())
    decimal_places = column_metadata.get('decimal_places', _get_decimal_places(values))
    is_integer = decimal_places == 0

    params = dd.UniformSamplerParams(
        low=float(low), high=float(high), decimal_places=decimal_places
    )
    return dd.SamplerColumnConfig(
        name=column_name,
        sampler_type=dd.SamplerType.UNIFORM,
        params=params,
        convert_to='int' if is_integer else None,
    )


def _create_datetime_config(dd, column_name, column_data, column_metadata):
    """Map a datetime column to a ``datetime`` sampler."""
    datetime_format = column_metadata.get('datetime_format')
    values = pd.to_datetime(column_data, format=datetime_format, errors='coerce').dropna()
    start = pd.to_datetime(column_metadata.get('range_min', values.min()), format=datetime_format)
    end = pd.to_datetime(column_metadata.get('range_max', values.max()), format=datetime_format)
    if pd.isna(start) or pd.isna(end):
        LOGGER.warning(
            f"Column '{column_name}' has no parseable datetimes, using a placeholder sampler."
        )
        return _create_placeholder_config(dd, column_name)

    output_format = datetime_format or '%Y-%m-%d %H:%M:%S'
    params = dd.DatetimeSamplerParams(
        start=start.strftime(output_format), end=end.strftime(output_format)
    )
    return dd.SamplerColumnConfig(
        name=column_name, sampler_type=dd.SamplerType.DATETIME, params=params
    )


def _create_id_config(dd, column_name):
    """Map an id column to a ``uuid`` sampler."""
    return dd.SamplerColumnConfig(
        name=column_name, sampler_type=dd.SamplerType.UUID, params=dd.UUIDSamplerParams()
    )


def _create_placeholder_config(dd, column_name):
    """Map unknown or unsupported pii columns to short anonymized placeholders."""
    params = dd.UUIDSamplerParams(prefix=PII_PLACEHOLDER_PREFIX, short_form=True)
    return dd.SamplerColumnConfig(name=column_name, sampler_type=dd.SamplerType.UUID, params=params)


def _create_text_config(dd, column_name, column_data, model_alias, context_columns):
    """Map a text column to an LLM generated column conditioned on the rest of the row."""
    examples = []
    for value in column_data.dropna().astype(str).unique()[:MAX_TEXT_EXAMPLES]:
        examples.append(repr(value[:MAX_TEXT_EXAMPLE_LENGTH]))

    prompt = f"Generate a realistic value for the column '{column_name}' of a tabular dataset."
    if examples:
        prompt += f' Example values from this column: {", ".join(examples)}.'

    references = [name for name in context_columns if name.isidentifier()]
    if references:
        row = ', '.join(f'{name}: {{{{ {name} }}}}' for name in references)
        prompt += f' The value must be consistent with the other values in this row: {row}.'

    prompt += ' Respond with the value only, without any explanation.'
    return dd.LLMTextColumnConfig(name=column_name, prompt=prompt, model_alias=model_alias)


def get_missing_value_proportions(data, metadata, table_name=None):
    """Return the proportion of missing values per column, honoring ``range_is_nullable``.

    Data Designer cannot generate missing values, so these proportions must be applied to the
    sampled data afterwards. Columns whose metadata says ``range_is_nullable: false`` are
    reported as having no missing values regardless of the data.

    Args:
        data (pd.DataFrame):
            The real data.
        metadata (sdv.metadata.Metadata or dict):
            Metadata V2 describing ``data``.
        table_name (str or None):
            The table to use when the metadata has more than one table.

    Returns:
        dict[str, float]:
            Mapping of column name to proportion of missing values in ``[0, 1]``.
    """
    table_metadata = _get_table_metadata(metadata, table_name)
    return {
        column_name: _get_missing_proportion(data[column_name], column_metadata)
        for column_name, column_metadata in table_metadata.columns.items()
        if column_name in data.columns
    }


def create_data_designer_config(
    data, metadata, table_name=None, model_alias=DEFAULT_MODEL_ALIAS, model_configs=None
):
    """Map a single table dataset and its Metadata V2 to a Data Designer config builder.

    Args:
        data (pd.DataFrame):
            The real data. Used to fit the sampler parameters that the metadata does not
            provide (frequencies, ranges, boolean rates) and to pick text examples.
        metadata (sdv.metadata.Metadata or dict):
            Metadata V2 describing ``data``. ``range_values``, ``range_min``, ``range_max``,
            ``decimal_places`` and ``range_is_nullable`` are used when present.
        table_name (str or None):
            The table to map when the metadata has more than one table. Defaults to the only
            table.
        model_alias (str):
            The model alias used by LLM generated (``text``) columns. Defaults to ``'text'``.
        model_configs (list[ModelConfig] or None):
            Model configurations to register with the builder. If ``None``, a single
            ``ModelConfig`` for ``model_alias`` is created that uses the NVIDIA provider.

    Returns:
        data_designer.config.DataDesignerConfigBuilder:
            A builder holding one column config per metadata column, ready to be passed to
            ``DataDesigner.create`` or ``.preview``.
    """
    dd = _import_data_designer()
    table_metadata = _get_table_metadata(metadata, table_name)
    if model_configs is None:
        model_configs = [_create_default_model_config(dd, model_alias)]

    missing = [name for name in table_metadata.columns if name not in data.columns]
    if missing:
        raise ValueError(f'The following columns are missing from the data: {missing}')

    context_columns = [
        name
        for name, column in table_metadata.columns.items()
        if column['sdtype'] in TEXT_CONTEXT_SDTYPES
    ]

    column_configs = []
    for column_name, column_metadata in table_metadata.columns.items():
        sdtype = column_metadata['sdtype']
        column_data = data[column_name]
        if sdtype == 'boolean':
            config = _create_boolean_config(dd, column_name, column_data)
        elif sdtype == 'numerical':
            config = _create_numerical_config(dd, column_name, column_data, column_metadata)
        elif sdtype == 'datetime':
            config = _create_datetime_config(dd, column_name, column_data, column_metadata)
        elif sdtype == 'id':
            config = _create_id_config(dd, column_name)
        elif sdtype == 'text':
            config = _create_text_config(dd, column_name, column_data, model_alias, context_columns)
        else:
            config = _create_categorical_config(dd, column_name, column_data, column_metadata)

        _attach_missing_proportion(config, _get_missing_proportion(column_data, column_metadata))
        column_configs.append(config)

    builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    for config in column_configs:
        builder.add_column(config)

    return builder


class DataDesignerSynthesizer(BaselineSynthesizer):
    """Custom wrapper for the DataDesigner synthesizer to make it work with SDGym."""

    LOGGER = logging.getLogger(__name__)
    _MODEL_KWARGS = None
    _MODALITY_FLAG = 'single_table'

    def _fit(self, data, metadata):
        config_builder = create_data_designer_config(data, metadata)
        try:
            from data_designer.interface import DataDesigner
        except Exception as exception:
            raise ValueError(
                "In order to use 'DataDesignerSynthesizer' you have to install the extra"
                " dependencies by running  pip install sdgym['data_designer'] "
            ) from exception

        model = DataDesigner()

        self._internal_synthesizer = model
        self._config_builder = config_builder

    def _sample_from_synthesizer(self, synthesizer, n_sample):
        """Sample synthetic data with specified sample count."""
        output = synthesizer._internal_synthesizer.create(
            synthesizer._config_builder, num_records=n_sample
        )
        return output.load_dataset()
