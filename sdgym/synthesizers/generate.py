"""Helpers to create SDGym synthesizer variants."""

from sdgym.synthesizers.base import (
    BaselineSynthesizer,
    MultiTableBaselineSynthesizer,
    _validate_modality,
)
from sdgym.synthesizers.utils import _get_supported_synthesizers


def create_synthesizer_variant(display_name, synthesizer_class, synthesizer_parameters):
    """Create a new synthesizer variant.

    Args:
        display_name (str):
            Name of this synthesizer, used for display purposes in results.
        synthesizer_class (str):
            Name of the SDV synthesizer class to wrap.
        synthesizer_parameters (dict):
            A dictionary of the parameter names and values that will be used for the synthesizer.

    Returns:
        class:
            The synthesizer class.
    """
    if synthesizer_class not in _get_supported_synthesizers():
        raise ValueError(f"Synthesizer '{synthesizer_class}' is not a SDGym supported synthesizer.")

    base_class = BaselineSynthesizer.get_subclasses().get(synthesizer_class)
    NewSynthesizer = type(
        f'Variant:{display_name}',
        (base_class,),
        {
            '__module__': __name__,
            '_MODEL_KWARGS': synthesizer_parameters,
            '_NATIVELY_SUPPORTED': False,
        },
    )

    return NewSynthesizer


def _create_synthesizer_class(display_name, get_trained_fn, sample_fn, modality):
    """Create a synthesizer class.

    Args:
        display_name(string):
            A string with the name of this synthesizer, used for display purposes only when
            the results are generated
        get_trained_synthesizer_fn (callable):
            A function to generate and train a synthesizer, given the real data and metadata.
        sample_from_synthesizer (callable):
            A function to sample from the given synthesizer.
        modality (str):
            The modality of the synthesizer. Either 'single_table' or 'multi_table'.

    Returns:
        class:
            The synthesizer class.
    """
    _validate_modality(modality)
    class_name = f'Custom:{display_name}'

    def get_trained_synthesizer(self, data, metadata):
        return get_trained_fn(data, metadata)

    if modality == 'multi_table':

        def sample_from_synthesizer(self, synthesizer, scale=1.0):
            return sample_fn(synthesizer, scale)

        base_class = MultiTableBaselineSynthesizer
    else:

        def sample_from_synthesizer(self, synthesizer, n_samples):
            return sample_fn(synthesizer, n_samples)

        base_class = BaselineSynthesizer

    CustomSynthesizer = type(
        class_name,
        (base_class,),
        {
            '__module__': __name__,
            '_NATIVELY_SUPPORTED': False,
            '_MODALITY_FLAG': modality,
            'get_trained_synthesizer': get_trained_synthesizer,
            'sample_from_synthesizer': sample_from_synthesizer,
        },
    )

    globals()[class_name] = CustomSynthesizer
    return CustomSynthesizer


def create_single_table_synthesizer(
    display_name, get_trained_synthesizer_fn, sample_from_synthesizer_fn
):
    """Create a single-table synthesizer class."""
    return _create_synthesizer_class(
        display_name,
        get_trained_synthesizer_fn,
        sample_from_synthesizer_fn,
        modality='single_table',
    )


def create_multi_table_synthesizer(
    display_name, get_trained_synthesizer_fn, sample_from_synthesizer_fn
):
    """Create a multi-table synthesizer class."""
    return _create_synthesizer_class(
        display_name,
        get_trained_synthesizer_fn,
        sample_from_synthesizer_fn,
        modality='multi_table',
    )
