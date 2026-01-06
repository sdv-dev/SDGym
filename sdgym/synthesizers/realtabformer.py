"""REaLTabFormer integration."""

import contextlib
import logging
from functools import partialmethod

import tqdm

from sdgym.synthesizers.base import BaselineSynthesizer


@contextlib.contextmanager
def prevent_tqdm_output():
    """Temporarily disables tqdm m."""
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    try:
        yield
    finally:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)


class RealTabFormerSynthesizer(BaselineSynthesizer):
    """Custom wrapper for the REaLTabFormer synthesizer to make it work with SDGym."""

    LOGGER = logging.getLogger(__name__)
    _MODEL_KWARGS = None

    def _get_trained_synthesizer(self, data, metadata):
        try:
            from realtabformer import REaLTabFormer
        except Exception as exception:
            raise ValueError(
                "In order to use 'RealTabFormerSynthesizer' you have to install the extra"
                " dependencies by running  pip install sdgym['realtabformer'] "
            ) from exception

        with prevent_tqdm_output():
            model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
            model = REaLTabFormer(model_type='tabular', **model_kwargs)

            # RealTabFormer >=0.2.3 changed the default behavior of `fit` by introducing
            # `save_full_every_epoch` and `gen_kwargs`. The new defaults break the SDGym
            # end-to-end test, so we set them explicitly to preserve the previous behavior.
            model.fit(data, save_full_every_epoch=0, gen_kwargs={})

        return model

    def _sample_from_synthesizer(self, synthesizer, n_sample):
        """Sample synthetic data with specified sample count."""
        return synthesizer.sample(n_sample)
