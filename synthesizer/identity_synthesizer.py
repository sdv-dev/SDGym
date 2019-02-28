from synthesizer_base import Synthesizer

class IdentitySynthesizer(Synthesizer):
    """docstring for IdentitySynthesizer."""

    def train(self, train_data, meta):
        self.learned = train_data

    def generate(self, n, meta):
        pass
