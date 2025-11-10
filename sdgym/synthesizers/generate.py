"""Helpers to create SDV synthesizer variants."""

from sdgym.synthesizers._sdv_lookup import find_sdv_synthesizer
from sdgym.synthesizers.base import BaselineSynthesizer
from sdgym.synthesizers.realtabformer import RealTabFormerSynthesizer
from sdgym.synthesizers.sdv import SDVMultiTableBaseline, SDVSingleTableBaseline

SYNTHESIZER_MAPPING = {
    'RealTabFormerSynthesizer': RealTabFormerSynthesizer,
}


def _list_available_synthesizers():
    import sdgym.synthesizers as synth_pkg

    out = []
    for name in dir(synth_pkg):
        obj = getattr(synth_pkg, name)
        if isinstance(obj, type) and issubclass(obj, BaselineSynthesizer):
            if obj in [BaselineSynthesizer, SDVSingleTableBaseline, SDVMultiTableBaseline]:
                continue

            out.append(name)

    return sorted(out)


def create_sdv_synthesizer_variant(display_name, synthesizer_class, synthesizer_parameters):
    """Create a new SDV synthesizer variant.

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
    try:
        sdv_cls, synthesizer_type = find_sdv_synthesizer(synthesizer_class)
    except KeyError:
        available_synthesizers = "', '".join(_list_available_synthesizers())
        raise ValueError(
            f'Synthesizer class {synthesizer_class!r} is not recognized. '
            f"Available SDV synthesizers: '{available_synthesizers}'"
        )

    if synthesizer_type == 'single_table':
        baseclass = SDVSingleTableBaseline
    else:
        baseclass = SDVMultiTableBaseline

    class NewSynthesizer(baseclass):
        """New Synthesizer class.

        Attributes:
            _SDV_CLASS:
                The SDV synthesizer class to wrap.
            _MODEL_KWARGS:
                The parameters to use when instantiating the SDV synthesizer.
        """

        _SDV_CLASS = sdv_cls
        _MODEL_KWARGS = synthesizer_parameters

    cls_name = f'Variant:{display_name}'
    NewSynthesizer.__name__ = cls_name
    NewSynthesizer.__module__ = __name__
    globals()[cls_name] = NewSynthesizer
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
