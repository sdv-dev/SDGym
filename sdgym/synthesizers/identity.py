import copy

from sdgym.synthesizers.base import BaselineSynthesizer


class IdentitySynthesizer(BaselineSynthesizer):
    """Trivial synthesizer.

    Returns the same exact data that is used to fit it.
    """

    def __init__(self):
        self._data = None

    def get_trained_synthesizer(self, data, metadata):
        self._data = data
        return None

    def sample_synthesizer(self, synthesizer, n_samples):
        return copy.deepcopy(self._data)
