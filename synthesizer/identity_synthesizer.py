from synthesizer_base import SynthesizerBase, run

class IdentitySynthesizer(SynthesizerBase):
    """docstring for IdentitySynthesizer."""

    def train(self, train_data):
        self.learned = train_data

    def generate(self, n):
        assert len(self.learned) == n
        return [(0, self.learned)]

    def init(self, meta, working_dir):
        pass


if __name__ == "__main__":
    run(IdentitySynthesizer())
