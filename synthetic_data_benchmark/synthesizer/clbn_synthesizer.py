from .synthesizer_base import SynthesizerBase, run
from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable
from .synthesizer_utils import DiscretizeTransformer
import json
import numpy as np

def bnsample(model, n):
    nodes_parents = model.structure
    processing_order = []

    while len(processing_order) != len(nodes_parents):
        update = False

        for id_, parents in enumerate(nodes_parents):
            if id_ in processing_order:
                continue

            flag = True
            for parent in parents:
                if not parent in processing_order:
                    flag = False

            if flag:
                processing_order.append(id_)
                update = True
        assert update


    data = np.zeros((n, len(nodes_parents)), dtype='int32')
    for current in processing_order:
        distribution = model.states[current].distribution
        if type(distribution) == DiscreteDistribution:
            data[:, current] = distribution.sample(n)
        else:
            assert type(distribution) == ConditionalProbabilityTable
            output_size = list(distribution.keys())
            output_size = max([int(x) for x in output_size]) + 1

            distribution = json.loads(distribution.to_json())
            distribution = distribution['table']

            distribution_dict = {}

            for row in distribution:
                key = tuple(np.asarray(row[:-2], dtype='int'))
                output = int(row[-2])
                p = float(row[-1])

                if not key in distribution_dict:
                    distribution_dict[key] = np.zeros(output_size)
                distribution_dict[key][int(output)] = p

            parents = nodes_parents[current]
            conds = data[:, parents]
            for _id, cond in enumerate(conds):
                data[_id, current] = np.random.choice(np.arange(output_size), p = distribution_dict[tuple(cond)])

    return data




class CLBNSynthesizer(SynthesizerBase):
    """docstring for IdentitySynthesizer."""

    def train(self, train_data):
        self.discretizer = DiscretizeTransformer(self.meta, 15)
        self.discretizer.fit(train_data)
        train_data_d = self.discretizer.transform(train_data)
        self.model = BayesianNetwork.from_samples(train_data_d, algorithm='chow-liu')

    def generate(self, n):
        data = bnsample(self.model, n)
        data = self.discretizer.inverse_transform(data)

        return [(0, data)]

    def init(self, meta, working_dir):
        self.meta = meta


if __name__ == "__main__":
    run(CLBNSynthesizer())
