"""Synthesizers module."""

from sdv.lite import TabularPreset
from sdv.relational import HMA1
from sdv.tabular import CTGAN, TVAE, CopulaGAN, GaussianCopula
from sdv.timeseries import PAR

from sdgym.synthesizers.base import (
    BaselineSynthesizer, MultiSingleTableBaselineSynthesizer, SingleTableBaselineSynthesizer)
from sdgym.synthesizers.sdv import FastMLPreset, SDVRelationalSynthesizer, SDVTabularSynthesizer

SYNTHESIZER_MAPPING = {
    'FastMLPreset': TabularPreset,
    'GaussianCopulaSynthesizer': GaussianCopula,
    'CTGANSynthesizer': CTGAN,
    'CopulaGANSynthesizer': CopulaGAN,
    'TVAESynthesizer': TVAE,
    'PARSynthesizer': PAR,
    'HMASynthesizer': HMA1,
}


def create_sdv_synthesizer_variant(display_name, synthesizer_class, synthesizer_parameters):
    """Create a new synthesizer that is a variant of an SDV tabular synthesizer.

    Args:
        display_name (string):
            A string with the name of this synthesizer, used for display purposes only
            when the results are generated.
        synthesizer_class (string):
            The name of the SDV synthesizer class. The available options are:

                * 'FastMLPreset'
                * 'GaussianCopulaSynthesizer'
                * 'CTGANSynthesizer',
                * 'CopulaGANSynthesizer'
                * 'TVAESynthesizer'
                * 'PARSynthesizer'
                * 'HMASynthesizer'

        synthesizer_parameters (dict):
            A dictionary of the parameter names and values that will be used for the synthesizer.

    Returns:
        class:
            The synthesizer class.
    """
    if synthesizer_class not in SYNTHESIZER_MAPPING.keys():
        raise ValueError(
            f'Synthesizer class {synthesizer_class} is not recognized. '
            f"The supported options are {', '.join(SYNTHESIZER_MAPPING.keys())}"
        )

    baseclass = SDVTabularSynthesizer
    if synthesizer_class == 'HMASynthesizer':
        baseclass = SDVRelationalSynthesizer
    if synthesizer_class == 'FastMLPreset':
        baseclass = FastMLPreset

    class NewSynthesizer(baseclass):
        """New Synthesizer class.

        Args:
            synthesizer_class (string):
                The name of the SDV synthesizer class. The available options are:

                    * 'FastMLPreset'
                    * 'GaussianCopulaSynthesizer'
                    * 'CTGANSynthesizer'
                    * 'CopulaGANSynthesizer'
                    * 'TVAESynthesizer'
                    * 'PARSynthesizer'

            synthesizer_parameters (dict):
                A dictionary of the parameter names and values that will be used for
                the synthesizer.
        """

        _MODEL = SYNTHESIZER_MAPPING.get(synthesizer_class)
        _MODEL_KWARGS = synthesizer_parameters

    NewSynthesizer.__name__ = f'Variant:{display_name}'

    return NewSynthesizer


def create_single_table_synthesizer(display_name, get_trained_synthesizer_fn,
                                    sample_from_synthesizer_fn):
    """Create a new single-table synthesizer.

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
    class NewSynthesizer(SingleTableBaselineSynthesizer):
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
                data (pandas.DataFrame):
                    The real data.
                metadata (sdv.Metadata):
                    The single table metadata.

            Returns:
                obj:
                    The trained synthesizer.
            """
            return get_trained_synthesizer_fn(data, metadata)

        def sample_from_synthesizer(self, synthesizer, num_samples):
            """Sample the desired number of samples from the given synthesizer.

            Args:
                synthesizer (obj):
                    The trained synthesizer.
                num_samples (int):
                    The number of samples to generate.

            Returns:
                pandas.DataFrame:
                    The synthetic data.
            """
            return sample_from_synthesizer_fn(synthesizer, num_samples)

    NewSynthesizer.__name__ = f'Custom:{display_name}'

    return NewSynthesizer


def create_multi_table_synthesizer(display_name, get_trained_synthesizer_fn,
                                   sample_from_synthesizer_fn):
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
                metadata (sdv.Metadata):
                    The multi table metadata.

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


def create_sequential_synthesizer(display_name, get_trained_synthesizer_fn,
                                  sample_from_synthesizer_fn):
    """Create a new sequential synthesizer.

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
    class NewSynthesizer(BaselineSynthesizer):
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
                metadata (sdv.Metadata):
                    The multi table metadata.

            Returns:
                obj:
                    The trained synthesizer.
            """
            return get_trained_synthesizer_fn(data, metadata)

        def sample_from_synthesizer(self, synthesizer, n_sequences):
            """Sample from the given synthesizer.

            Args:
                synthesizer (obj):
                    The trained synthesizer.
                n_sequences (int):
                    The number of sequences to generate.

            Returns:
                dict:
                    The synthetic data. A mapping of table names to table data.
            """
            return sample_from_synthesizer_fn(synthesizer, n_sequences)

    NewSynthesizer.__name__ = f'Custom:{display_name}'

    return NewSynthesizer
