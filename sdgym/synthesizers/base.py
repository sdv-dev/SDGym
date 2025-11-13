"""Base classes for synthesizers."""

import abc
import logging
import warnings

from sdv.metadata import Metadata

LOGGER = logging.getLogger(__name__)


class BaselineSynthesizer(abc.ABC):
    """Base class for all the ``SDGym`` baselines."""

    _MODEL_KWARGS = {}
    _NATIVELY_SUPPORTED = True

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
    def _get_supported_synthesizers(cls):
        """Get the natively supported synthesizer class names."""
        subclasses = cls.get_subclasses(include_parents=True)
        synthesizers = set()
        for name, subclass in subclasses.items():
            if subclass._NATIVELY_SUPPORTED:
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
