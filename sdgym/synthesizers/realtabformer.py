from realtabformer import REaLTabFormer

from sdgym.synthesizers.base import BaselineSynthesizer


class REaLTabFormerSynthesizer(BaselineSynthesizer):
    """Custom wrapper for the REaLTabFormer synthesizer to make it work with SDGym."""

    def __init__(self, **kwargs):
        self.model = REaLTabFormer(model_type="tabular", **kwargs)

    def fit(self, data):
        """Fit the REaLTabFormer model on the provided dataset."""
        self.model.fit(data)

    def sample(self, n_samples):
        """Generate synthetic data samples."""
        return self.model.sample(n_samples)

    def save(self, path):
        """Save the model to a given directory."""
        self.model.save(path)

    @classmethod
    def load(cls, path):
        """Load a previously saved model from a directory."""
        model = REaLTabFormer.load_from_dir(path)
        instance = cls()
        instance.model = model
        return instance

    def _get_trained_synthesizer(self, data, metadata):
        self.model.fit(data, device="cpu")
        return self

    def _sample_from_synthesizer(self, synthesizer, n_sample):
        """Sample synthetic data with specified sample count."""
        return synthesizer.sample(n_sample)

    def sample(self, n_samples):
        """Generate synthetic data samples."""
        if self.model is None:
            raise ValueError("Model is not trained. Call `fit` before sampling.")
        return self.model.sample(n_samples, device="cpu")