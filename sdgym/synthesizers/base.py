import logging

LOGGER = logging.getLogger(__name__)


class BaseSynthesizer:
    """Base class for all default synthesizers of ``SDGym``."""

    def fit(self, data, categoricals=tuple(), ordinals=tuple()):
        pass

    def sample(self, samples):
        pass

    def fit_sample(self, data, categoricals=tuple(), ordinals=tuple()):
        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit(data, categoricals, ordinals)

        LOGGER.info("Sampling %s", self.__class__.__name__)
        return self.sample(data.shape[0])
