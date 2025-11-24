"""DataIdentity module."""

import copy

from sdgym.synthesizers.base import BaselineSynthesizer


class DataIdentity(BaselineSynthesizer):
    """Trivial synthesizer.

    Returns the same exact data that is used to fit it.
    """

    _MODALITY_FLAG = 'single_table'

    def __init__(self):
        self._data = None

    def _fit(self, data, metadata):
        """Fit the synthesizer to the data.

        Args:
            data (pandas.DataFrame):
                The data to fit the synthesizer to.
            metadata (dict):
                The metadata dictionary.
        """
        self._data = data

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
        return copy.deepcopy(synthesizer._data)
