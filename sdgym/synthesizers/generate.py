"""Helpers to create SDGym synthesizer variants."""

from sdgym.synthesizers.base import BaselineSynthesizer
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
            '_NATIVELY_SUPPORTED': False,
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
