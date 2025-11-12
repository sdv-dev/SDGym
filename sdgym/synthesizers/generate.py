"""Helpers to create SDGym synthesizer variants."""

from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.synthesizers.realtabformer import RealTabFormerSynthesizer
from sdgym.synthesizers.sdv import (
    BaselineSDVSynthesizer,
    _get_all_sdv_synthesizers,
    _get_sdv_synthesizers,
)

SYNTHESIZER_MAPPING = {
    'RealTabFormerSynthesizer': RealTabFormerSynthesizer,
}


def create_synthesizer_variant(display_name, synthesizer_class, synthesizer_parameters):
    """Create a new synthesizer variant.

    Args:
        display_name (str):
            Name of this synthesizer, used for display purposes in results.
        synthesizer_class (str):
            Name of the SDV synthesizer class to wrap.
        synthesizer_parameters (dict):
            A dictionary of the parameter names and values that will be used for the synthesizer.
        modality (str):
            The modality of the synthesizer, either 'single_table' or 'multi_table'.
            Defaults to 'single_table'.

    Returns:
        class:
            The synthesizer class.
    """
    base_class = SYNTHESIZER_MAPPING.get(synthesizer_class)
    if base_class:

        class NewSynthesizer(base_class):
            _MODEL = SYNTHESIZER_MAPPING.get(synthesizer_class)
            _MODEL_KWARGS = synthesizer_parameters

        NewSynthesizer.__name__ = f'Variant:{display_name}'
        return NewSynthesizer

    all_sdv_synthesizers = _get_all_sdv_synthesizers()
    if synthesizer_class not in all_sdv_synthesizers:
        raise ValueError(f"Synthesizer '{synthesizer_class}' is not a SDV synthesizer.")

    modality = (
        'single_table'
        if synthesizer_class in _get_sdv_synthesizers('single_table')
        else 'multi_table'
    )
    synthesizer = BaselineSDVSynthesizer(synthesizer_class, modality, synthesizer_parameters)
    synthesizer.__name__ = f'Variant:{display_name}'

    return synthesizer


def _create_synthesizer_class(display_name, get_trained_fn, sample_fn, sample_arg_name):
    """Create a synthesizer class.

    Args:
        display_name(string):
            A string with the name of this synthesizer, used for display purposes only when
            the results are generated
        get_trained_synthesizer_fn (callable):
            A function to generate and train a synthesizer, given the real data and metadata.
        sample_from_synthesizer (callable):
            A function to sample from the given synthesizer.
        sample_arg_name (str):
            The name of the argument used to specify the number of samples to generate.
            Either 'num_samples' for single-table synthesizers, or 'scale' for multi-table
            synthesizers.

    Returns:
        class:
            The synthesizer class.
    """
    class_name = f'Custom:{display_name}'

    def get_trained_synthesizer(self, data, metadata):
        return get_trained_fn(data, metadata)

    if sample_arg_name == 'num_samples':

        def sample_from_synthesizer(self, synthesizer, num_samples):
            return sample_fn(synthesizer, num_samples)

    else:

        def sample_from_synthesizer(self, synthesizer, scale):
            return sample_fn(synthesizer, scale)

    CustomSynthesizer = type(
        class_name,
        (BaselineSynthesizer,),
        {
            '__module__': __name__,
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
        sample_arg_name='num_samples',
    )


def create_multi_table_synthesizer(
    display_name, get_trained_synthesizer_fn, sample_from_synthesizer_fn
):
    """Create a multi-table synthesizer class."""
    return _create_synthesizer_class(
        display_name,
        get_trained_synthesizer_fn,
        sample_from_synthesizer_fn,
        sample_arg_name='scale',
    )
