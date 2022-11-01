import copy

from sdgym.synthesizers.base import BaselineSynthesizer


class IdentitySynthesizer(BaselineSynthesizer):
    """Trivial synthesizer.

    Returns the same exact data that is used to fit it.
    """

    def fit_sample(self, real_data, metadata):
        del metadata
        return copy.deepcopy(real_data)
