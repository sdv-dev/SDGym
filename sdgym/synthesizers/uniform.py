from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer

from sdgym.synthesizers.base import SingleTableBaselineSynthesizer


class UniformSynthesizer(SingleTableBaselineSynthesizer):
    """Synthesizer that samples each column using a Uniform distribution."""

    def _get_trained_synthesizer(self, real_data, metadata):
        metadata = SingleTableMetadata.load_from_dict(metadata)
        synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution='uniform')
        synthesizer.fit(real_data)

        return synthesizer

    def _sample_from_synthesizer(self, synthesizer, n_samples):
        return synthesizer.sample(n_samples)
