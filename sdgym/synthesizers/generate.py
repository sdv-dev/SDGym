"""Helpers to create SDV synthesizer variants."""

from sdgym.synthesizers._sdv_dynamic import (
    SDVMultiTableBaseline,
    SDVSingleTableBaseline,
)
from sdgym.synthesizers._sdv_lookup import find_sdv_synthesizer
from sdgym.synthesizers.base import BaselineSynthesizer, MultiSingleTableBaselineSynthesizer
from sdgym.synthesizers.realtabformer import RealTabFormerSynthesizer

SYNTHESIZER_MAPPING = {
    'RealTabFormerSynthesizer': RealTabFormerSynthesizer,
}


def _list_available_synthesizers():
    import sdgym.synthesizers as synth_pkg

    out = []
    for name in dir(synth_pkg):
        obj = getattr(synth_pkg, name)
        if isinstance(obj, type) and issubclass(obj, BaselineSynthesizer):
            if obj is BaselineSynthesizer:
                continue
            out.append(name)

    return sorted(out)


def create_sdv_synthesizer_variant(display_name, synthesizer_class, synthesizer_parameters):
    """Create a new SDV synthesizer variant."""
    try:
        sdv_cls, kind = find_sdv_synthesizer(synthesizer_class)
    except KeyError:
        available_synthesizers = "', '".join(_list_available_synthesizers())
        raise ValueError(
            f'Synthesizer class {synthesizer_class!r} is not recognized. '
            f"Available SDV synthesizers: '{available_synthesizers}'"
        )

    if kind == 'single_table':
        baseclass = SDVSingleTableBaseline
    else:
        baseclass = SDVMultiTableBaseline

    class NewSynthesizer(baseclass):
        _SDV_CLASS = sdv_cls
        _BASE_SYNTHESIZER_CLASS = sdv_cls
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
        display_name (str):
            Name of this synthesizer, used for display purposes in results.
        get_trained_synthesizer_fn (callable):
            Function that, given (data, metadata), returns a trained synthesizer.
        sample_from_synthesizer_fn (callable):
            Function that, given (synthesizer, n), returns sampled data.

    Returns:
        type: the synthesizer class.
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

    # make it importable/picklable
    CustomSynthesizer.__name__ = class_name
    CustomSynthesizer.__module__ = __name__
    globals()[class_name] = CustomSynthesizer

    return CustomSynthesizer


def create_multi_table_synthesizer(
    display_name, get_trained_synthesizer_fn, sample_from_synthesizer_fn
):
    """Create a new multi-table synthesizer.

    Args:
        display_name(string):
            A string with the name of this synthesizer, used for display purposes only when
            the results are generated
        get_trained_synthesizer_fn (callable):
            A function to generate and train a synthesizer, given the real data and metadata.
        sample_from_synthesizer (callable):
            A function to sample from the given synthesizer.

    Returns:
        class:
            The synthesizer class.
    """

    class NewSynthesizer(MultiSingleTableBaselineSynthesizer):
        """New Synthesizer class.

        Args:
            get_trained_synthesizer_fn (callable):
                Function to replace the ``get_trained_synthesizer`` method.
            sample_from_synthesizer_fn (callable):
                Function to replace the ``sample_from_synthesizer`` method.
        """

        def get_trained_synthesizer(self, data, metadata):
            """Create and train a synthesizer, given the real data and metadata.

            Args:
                data (dict):
                    The real data. A mapping of table names to table data.
                metadata (dict):
                    The multi table metadata dictionary.

            Returns:
                obj:
                    The trained synthesizer.
            """
            return get_trained_synthesizer_fn(data, metadata)

        def sample_from_synthesizer(self, synthesizer):
            """Sample from the given synthesizer.

            Args:
                synthesizer (obj):
                    The trained synthesizer.

            Returns:
                dict:
                    The synthetic data. A mapping of table names to table data.
            """
            return sample_from_synthesizer_fn(synthesizer)

    NewSynthesizer.__name__ = f'Custom:{display_name}'

    return NewSynthesizer
