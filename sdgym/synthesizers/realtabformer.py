"""REaLTabFormer integration."""

import contextlib
import os
from functools import partialmethod

import torch
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
            print('PYTORCH_ENABLE_MPS_FALLBACK')
            print(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'])
            print('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
            print(os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'])
            print('<<<<<<<<<<<<<<<<<<MPS AVAILABLE FIT>>>>>>>>>>>>')
            print(torch.backends.mps.is_available())

        return model

    def _sample_from_synthesizer(self, synthesizer, n_sample):
        """Sample synthetic data with specified sample count."""
        print('PYTORCH_ENABLE_MPS_FALLBACK')
        print(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'])
        print('PYTORCH_MPS_HIGH_WATERMARK_RATIO')
        print(os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'])
        print('<<<<<<<<<<<<<<<<<<MPS AVAILABLE SAMPLE>>>>>>>>>>>>')
        print(torch.backends.mps.is_available())
        return synthesizer.sample(n_sample, device='cpu')
