from .synthesizer_base import SynthesizerBase, run
import numpy as np

class IdentitySynthesizer(SynthesizerBase):
    """docstring for IdentitySynthesizer."""

    def train(self, train_data):
        self.learned = train_data.copy()

    def generate(self, n):
        assert len(self.learned) >= n
        np.random.shuffle(self.learned)
        return [(0, self.learned[:n])]

    def init(self, meta, working_dir):
        pass


if __name__ == "__main__":
    run(IdentitySynthesizer())
