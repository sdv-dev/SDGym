"""REaLTabFormer integration."""

import contextlib
import dataclasses
import logging
from functools import partialmethod

import tqdm

from sdgym.synthesizers.base import BaselineSynthesizer

REMOVED_TRAINING_ARGS = ('overwrite_output_dir',)


@contextlib.contextmanager
def prevent_tqdm_output():
    """Temporarily disables tqdm m."""
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    try:
        yield
    finally:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)


def _remove_unsupported_training_args(model):
    """Drop deprecated args that are no longer used by transformers."""
    from transformers import TrainingArguments

    supported = {field.name for field in dataclasses.fields(TrainingArguments)}
    for name in REMOVED_TRAINING_ARGS:
        if name not in supported:
            model.training_args_kwargs.pop(name, None)


class RealTabFormerSynthesizer(BaselineSynthesizer):
    """Custom wrapper for the REaLTabFormer synthesizer to make it work with SDGym."""

    LOGGER = logging.getLogger(__name__)
    _MODEL_KWARGS = None
    _MODALITY_FLAG = 'single_table'

    def _fit(self, data, metadata):
        """Fit the REaLTabFormer model to the data."""
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
            _remove_unsupported_training_args(model)
            model.fit(data)

        self._internal_synthesizer = model

    def _sample_from_synthesizer(self, synthesizer, n_sample):
        """Sample synthetic data with specified sample count."""
        return synthesizer._internal_synthesizer.sample(n_sample)
