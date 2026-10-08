"""NVIDIA Data Designer integration."""

import logging
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sdv.metadata import Metadata

from sdgym.synthesizers.base import BaselineSynthesizer

LOGGER = logging.getLogger(__name__)

MIN_PYTHON_VERSION = (3, 10)
DEFAULT_MODEL_ALIAS = 'text'
DEFAULT_MODEL_ID = 'nvidia/nvidia-nemotron-nano-9b-v2'
DEFAULT_MODEL_PROVIDER = 'nvidia'
DEFAULT_TEMPERATURE = 0.85
DEFAULT_TOP_P = 0.95
PII_PLACEHOLDER_PREFIX = 'sdgym-pii-'
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
        raise ImportError('DataDesignerSynthesizer only supports python >= 3.10')

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
        raise TypeError(f'Expected sdv.Metadata, got {type(metadata).__name__}. ')

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


def _create_default_model_config(
        dd,
        model_alias,
        model=DEFAULT_MODEL_ID,
        provider=DEFAULT_MODEL_PROVIDER,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
    ):

    return dd.ModelConfig(
        alias=model_alias,
        model=model,
        provider=provider,
        inference_parameters=dd.ChatCompletionInferenceParams(temperature=temperature, top_p=top_p),
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
    """Map a numerical column to a ``uniform`` sampler."""
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
    """Map an id column to a ``uuid`` sampler, no other id sampler is available."""
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

    prompt = (
        'You are a helpful assistant that generates synthetic data.'
        f"Generate a realistic value for the column '{column_name}' of a tabular dataset."
    )
    if examples:
        prompt += f' Here are a few examples from this column: {", ".join(examples)}.'

    if context_columns:
        row = ', '.join(f'{{ {name} }}' for name in context_columns)
        prompt += f' Refer to the values in these columns for context: {row}.'

    prompt += ' Respond with the value only, without any explanation.'
    return dd.LLMTextColumnConfig(name=column_name, prompt=prompt, model_alias=model_alias)


def create_data_designer_config(
    data,
    metadata,
    table_name=None,
    model_alias=DEFAULT_MODEL_ALIAS,
    model=DEFAULT_MODEL_ID,
    provider=DEFAULT_MODEL_PROVIDER,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P
):
    """Map a single table dataset and its metadata to a Data Designer config builder.

    Args:
        data (pd.DataFrame):
            The real data. Used to fit the sampler parameters that the metadata does not
            provide (frequencies, ranges, boolean rates) and to pick text examples.
        metadata (sdv.Metadata):
            Metadata describing the data. If range values are not present, they are
            extracted directly from data.
        table_name (str or None):
            The table to map when the metadata has more than one table. Defaults to the only
            table.
        model_alias (str):
            The model alias used by LLM generated text columns. Defaults to 'text'.
        model (str):
            TODO: update model docstrings.
        provider (str):
            TODO: update model provider docstrings.
        temperature (float):
            Sampling temperature of the model used by LLM generated text columns. Higher
            values produce more varied text. Defaults to 0.85.
        top_p (float):
            Nucleus sampling probability of the model used by LLM generated text columns.
            Defaults to 0.95.

    Returns:
        DataDesignerConfigBuilder:
            A builder holding one column config per metadata column.
    """
    dd = _import_data_designer()
    table_metadata = _get_table_metadata(metadata, table_name)
    model_configs = [
        _create_default_model_config(dd, model_alias, model, provider, temperature, top_p)
    ]

    context_columns = [
        name
        for name, column_meta in table_metadata.columns.items()
        if column_meta['sdtype'] in TEXT_CONTEXT_SDTYPES
    ]

    column_configs = []
    for column_name, column_metadata in table_metadata.columns.items():
        sdtype = column_metadata['sdtype']
        column_data = data[column_name]
        if column_data.isna().all():
            config = _create_placeholder_config(dd, column_name)
        elif sdtype == 'boolean':
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
        metadata.validate_data(data)
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        self._artifact_path = model_kwargs.pop('artifact_path', None)
        self._cleanup_artifacts = model_kwargs.pop('cleanup_artifacts', True)
        self._config_builder = create_data_designer_config(data, metadata, **model_kwargs)

    def _sample_from_synthesizer(self, synthesizer, n_sample):
        """Sample synthetic data with specified sample count."""
        try:
            from data_designer.interface import DataDesigner
        except Exception as exception:
            raise ImportError(
                "To use 'DataDesignerSynthesizer' you have to install the extra "
                "dependencies by running pip install sdgym['data_designer']"
            ) from exception

        is_temporary = synthesizer._artifact_path is None
        if is_temporary:
            artifact_path = Path(tempfile.mkdtemp(prefix='sdgym-data-designer-'))
        else:
            artifact_path = Path(synthesizer._artifact_path)

        # A unique name keeps concurrent or repeated sample calls from sharing a folder.
        dataset_name = f'dataset-{uuid.uuid4().hex}'
        try:
            output = DataDesigner(artifact_path=artifact_path).create(
                synthesizer._config_builder, num_records=n_sample, dataset_name=dataset_name
            )
            return output.load_dataset()
        finally:
            if synthesizer._cleanup_artifacts:
                to_remove = artifact_path if is_temporary else artifact_path / dataset_name
                shutil.rmtree(to_remove, ignore_errors=True)
            else:
                self.LOGGER.info(f'Data Designer artifacts kept in {artifact_path / dataset_name}')
