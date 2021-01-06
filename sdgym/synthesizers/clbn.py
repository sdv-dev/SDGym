import json

import numpy as np
from pomegranate import BayesianNetwork, ConditionalProbabilityTable, DiscreteDistribution

from sdgym.synthesizers.base import LegacySingleTableBaseline
from sdgym.synthesizers.utils import DiscretizeTransformer


class CLBN(LegacySingleTableBaseline):
    """CLBNSynthesizer."""

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.discretizer = DiscretizeTransformer(n_bins=15)
        self.discretizer.fit(data, categorical_columns, ordinal_columns)
        discretized_data = self.discretizer.transform(data)
        self.model = BayesianNetwork.from_samples(discretized_data, algorithm='chow-liu')

    def bn_sample(self, num_samples):
        """Sample from the bayesian network.

        Args:
            num_samples(int): Number of samples to generate.
        """
        nodes_parents = self.model.structure
        processing_order = []

        while len(processing_order) != len(nodes_parents):
            update = False

            for id_, parents in enumerate(nodes_parents):
                if id_ in processing_order:
                    continue

                flag = True
                for parent in parents:
                    if parent not in processing_order:
                        flag = False

                if flag:
                    processing_order.append(id_)
                    update = True

            assert update

        data = np.zeros((num_samples, len(nodes_parents)), dtype='int32')
        for current in processing_order:
            distribution = self.model.states[current].distribution
            if isinstance(distribution, DiscreteDistribution):
                data[:, current] = distribution.sample(num_samples)
            else:
                assert isinstance(distribution, ConditionalProbabilityTable)
                output_size = list(distribution.keys())
                output_size = max([int(x) for x in output_size]) + 1

                distribution = json.loads(distribution.to_json())
                distribution = distribution['table']

                distribution_dict = {}

                for row in distribution:
                    key = tuple(np.asarray(row[:-2], dtype='int'))
                    output = int(row[-2])
                    p = float(row[-1])

                    if key not in distribution_dict:
                        distribution_dict[key] = np.zeros(output_size)

                    distribution_dict[key][int(output)] = p

                parents = nodes_parents[current]
                conds = data[:, parents]
                for _id, cond in enumerate(conds):
                    data[_id, current] = np.random.choice(
                        np.arange(output_size),
                        p=distribution_dict[tuple(cond)]
                    )

        return data

    def sample(self, num_samples):
        data = self.bn_sample(num_samples)
        return self.discretizer.inverse_transform(data)
