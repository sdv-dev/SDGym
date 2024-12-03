"""REaLTabFormer integration."""

import contextlib
import logging
import os
from functools import partialmethod

import torch
import tqdm

from sdgym.synthesizers.base import BaselineSynthesizer, LOGGER


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

    def _get_trained_synthesizer(self, data, metadata):
        try:
            from realtabformer import REaLTabFormer
        except Exception as exception:
            raise ValueError(
                "In order to use 'RealTabFormerSynthesizer' you have to install sdgym as "
                "sdgym['realtabformer']."
            ) from exception

        with prevent_tqdm_output():
            model = REaLTabFormer(model_type='tabular')
            model.fit(data, device='cpu')
            LOGGER.debug('PYTORCH_ENABLE_MPS_FALLBACK')
            LOGGER.debug(os.getenv('PYTORCH_ENABLE_MPS_FALLBACK') )
            LOGGER.debug('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
            LOGGER.debug(os.getenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO'))
            LOGGER.debug('<<<<<<<<<<<<<<<<<<MPS AVAILABLE FIT>>>>>>>>>>>>')
            LOGGER.debug(torch.backends.mps.is_available())

        return model

    def _sample_from_synthesizer(self, synthesizer, n_sample):
        """Sample synthetic data with specified sample count."""
        LOGGER.debug('PYTORCH_ENABLE_MPS_FALLBACK')
        LOGGER.debug(os.getenv('PYTORCH_ENABLE_MPS_FALLBACK'))
        LOGGER.debug('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
        LOGGER.debug(os.getenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO'))
        LOGGER.debug('<<<<<<<<<<<<<<<<<<MPS AVAILABLE SAMPLE>>>>>>>>>>>>')
        LOGGER.debug(torch.backends.mps.is_available())
        return synthesizer.sample(n_sample, device='cpu')
