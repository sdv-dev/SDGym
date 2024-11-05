"""REaLTabFormer integration."""

from sdgym.synthesizers.base import BaselineSynthesizer


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

        model = REaLTabFormer(model_type='tabular')
        model.fit(data, device='cpu')
        return model

    def _sample_from_synthesizer(self, synthesizer, n_sample):
        """Sample synthetic data with specified sample count."""
        return synthesizer.sample(n_sample)
