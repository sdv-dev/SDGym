"""Base classes for synthesizers."""

import abc
import logging
import warnings

from sdv.metadata import Metadata

LOGGER = logging.getLogger(__name__)


def _is_valid_modality(modality):
    return modality in ('single_table', 'multi_table')


def _validate_modality(modality):
    if not _is_valid_modality(modality):
        raise ValueError(
            f"Modality '{modality}' is not valid. Must be either 'single_table' or 'multi_table'."
        )


class BaselineSynthesizer(abc.ABC):
    """Base class for all the ``SDGym`` baselines."""

    _MODEL_KWARGS = {}
    _NATIVELY_SUPPORTED = True
    _MODALITY_FLAG = None

    @classmethod
    def get_subclasses(cls, include_parents=False):
        """Recursively find subclasses of this Baseline.

        Args:
            include_parents (bool):
                Whether to include subclasses which are parents to
                other classes. Defaults to ``False``.
        """
        subclasses = {}
        for child in cls.__subclasses__():
            grandchildren = child.get_subclasses(include_parents)
            subclasses.update(grandchildren)
            if include_parents or not grandchildren:
                subclasses[child.__name__] = child

        return subclasses

    @classmethod
    def _get_supported_synthesizers(cls, modality):
        """Get the natively supported synthesizer class names."""
        _validate_modality(modality)
        subclasses = cls.get_subclasses(include_parents=True)
        synthesizers = set()
        for name, subclass in subclasses.items():
            if subclass._NATIVELY_SUPPORTED and subclass._MODALITY_FLAG == modality:
                synthesizers.add(name)

        return sorted(synthesizers)

    @classmethod
    def get_baselines(cls):
        """Get baseline classes."""
        subclasses = cls.get_subclasses(include_parents=True)
        synthesizers = []
        for _, subclass in subclasses.items():
            if abc.ABC not in subclass.__bases__:
                synthesizers.append(subclass)

        return synthesizers

    def _validate_modality_flag(self):
        if not _is_valid_modality(self._MODALITY_FLAG):
            raise ValueError(
                f"The `_MODALITY_FLAG` '{self._MODALITY_FLAG}' of the synthesizer is not valid. "
                "Must be either 'single_table' or 'multi_table'."
            )

    def get_trained_synthesizer(self, data, metadata):
        """Get a synthesizer that has been trained on the provided data and metadata.

        Args:
            data (pandas.DataFrame):
                The data to train on.
            metadata (dict):
                The metadata dictionary.

        Returns:
            obj:
                The synthesizer object.
        """
        self._validate_modality_flag()
        metadata_object = Metadata()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            metadata = metadata_object.load_from_dict(metadata)

        return self._get_trained_synthesizer(data, metadata)

    def sample_from_synthesizer(self, synthesizer, n_samples):
        """Sample data from the provided synthesizer.

        Args:
            synthesizer (obj):
                The synthesizer object to sample data from.
            n_samples (int):
                The number of samples to create.

        Returns:
            pandas.DataFrame or dict:
                The sampled data. If single-table, should be a DataFrame. If multi-table,
                should be a dict mapping table name to DataFrame.
        """
        return self._sample_from_synthesizer(synthesizer, n_samples)
