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
        _SDV_CLASS = sdv_cls
        _MODEL_KWARGS = synthesizer_parameters

    cls_name = f'Variant:{display_name}'
    NewSynthesizer.__name__ = cls_name
    NewSynthesizer.__module__ = __name__
    globals()[cls_name] = NewSynthesizer
    return NewSynthesizer


def create_single_table_synthesizer(
    display_name, get_trained_synthesizer_fn, sample_from_synthesizer_fn
):
    """Create a new single-table synthesizer.

    Args:
        display_name(string):
            A string with the name of this synthesizer, used for display purposes only when
            the results are generated
        get_trained_synthesizer_fn (callable):
            A function to generate and train a synthesizer, given the real data and metadata.
        sample_from_synthesizer_fn (callable):
            A function to sample from the given synthesizer.

    Returns:
        class:
            The synthesizer class.
    """

    class NewSynthesizer(BaselineSynthesizer):
        def get_trained_synthesizer(self, data, metadata):
            return self.synthesizer_fn['get_trained_synthesizer_fn'](data, metadata)

        def sample_from_synthesizer(self, synthesizer, num_samples):
            return self.synthesizer_fn['sample_from_synthesizer_fn'](synthesizer, num_samples)

    class_name = f'Custom:{display_name}'

    CustomSynthesizer = type(
        class_name,
        (NewSynthesizer,),
        {
            'synthesizer_fn': {
                'get_trained_synthesizer_fn': get_trained_synthesizer_fn,
                'sample_from_synthesizer_fn': sample_from_synthesizer_fn,
            },
        },
    )

    CustomSynthesizer.__name__ = class_name
    CustomSynthesizer.__module__ = __name__
    globals()[class_name] = CustomSynthesizer

    return CustomSynthesizer
