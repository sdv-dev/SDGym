import argparse
import json
import os
import pandas as pd
import numpy as np

from pomegranate import *

class MultivariateMaker(object):
    """base class for simulated bayesian network"""

    def __init__(self, dist_type):
        self.model = None

    def sample(self, n):
        nodes_parents = self.model.structure
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

        data = np.empty((n, len(nodes_parents)), dtype='str')
        for current in processing_order:
            distribution = self.model.states[current].distribution
            if type(distribution) == DiscreteDistribution:
                data[:, current] = distribution.sample(n)
            else:
                assert type(distribution) == ConditionalProbabilityTable
                parents_map = nodes_parents[current]
                parents = distribution.parents
                for _id in range(n):
                    values = {}
                    for i in range(len(parents_map)):
                        values[parents[i]] = data[_id, parents_map[i]]
                    data[_id, current] = distribution.sample(parent_values=values)

        return data




class ChainMaker(MultivariateMaker):

    def __init__(self):
        A = DiscreteDistribution({'1': 1./3, '2': 1./3, '3': 1./3})
        B = ConditionalProbabilityTable(
            [['1','1',0.5],
            ['1','2',0.5],
            ['1','3',0],
            ['2','1',0],
            ['2','2',0.5],
            ['2','3',0.5],
            ['3','1',0.5],
            ['3','2',0],
            ['3','3',0.5],
            ],[A])
        C = ConditionalProbabilityTable(
            [['1','1',0.5],
            ['1','2',0.5],
            ['1','3',0],
            ['2','1',0],
            ['2','2',0.5],
            ['2','3',0.5],
            ['3','1',0.5],
            ['3','2',0],
            ['3','3',0.5],
            ],[B])

        s1 = Node(A, name="A")
        s2 = Node(B, name="B")
        s3 = Node(C, name="C")

        model = BayesianNetwork("ChainSampler")
        model.add_states(s1, s2, s3)
        model.add_edge(s1, s2)
        model.add_edge(s2, s3)
        model.bake()
        self.model = model

        meta = []
        for i in range(self.model.node_count()):
            meta.append({
                "name": None,
                "type": "Categorical",
                "size": 3,
                "i2s": ['1', '2', '3']
        })
        self.meta = meta


class TreeMaker(MultivariateMaker):
    def __init__(self):
        A = DiscreteDistribution({'1': 1./3, '2': 1./3, '3': 1./3})
        B = ConditionalProbabilityTable(
            [['1','1',0.5],
            ['1','2',0.5],
            ['1','3',0],
            ['2','1',0],
            ['2','2',0.5],
            ['2','3',0.5],
            ['3','1',0.5],
            ['3','2',0],
            ['3','3',0.5],
            ],[A])
        C = ConditionalProbabilityTable(
            [['1','4',0.5],
            ['1','5',0.5],
            ['1','6',0],
            ['2','4',0],
            ['2','5',0.5],
            ['2','6',0.5],
            ['3','4',0.5],
            ['3','5',0],
            ['3','6',0.5],
            ],[A])

        s1 = Node(A, name="A")
        s2 = Node(B, name="B")
        s3 = Node(C, name="C")

        model = BayesianNetwork("tree")
        model.add_states(s1, s2, s3)
        model.add_edge(s1, s2)
        model.add_edge(s1, s3)
        model.bake()
        self.model = model

        meta = []
        for i in range(self.model.node_count()-1):
            meta.append({
                "name": None,
                "type": "Categorical",
                "size": 3,
                "i2s": ['1', '2', '3']
        })
        meta.append({
                "name": None,
                "type": "Categorical",
                "size": 3,
                "i2s": ['4', '5', '6']
        })
        self.meta = meta

class FCMaker(MultivariateMaker):
    def __init__(self):
        Rain = DiscreteDistribution({'T': 0.2, 'F': 0.8})
        Sprinkler = ConditionalProbabilityTable(
            [['F','T',0.4],
            ['F','F',0.6],
            ['T','T',0.1],
            ['T','F',0.9],
            ],[Rain])
        Wet = ConditionalProbabilityTable(
            [['F','F','T',0.01],
            ['F','F','F',0.99],
            ['F','T','T',0.8],
            ['F','T','F',0.2],
            ['T','F','T',0.9],
            ['T','F','F',0.1],
            ['T','T','T',0.99],
            ['T','T','F',0.01],
            ],[Sprinkler,Rain])

        s1 = Node(Rain, name="Rain")
        s2 = Node(Sprinkler, name="Sprinkler")
        s3 = Node(Wet, name="Wet")

        model = BayesianNetwork("Simple fully connected")
        model.add_states(s1, s2, s3)
        model.add_edge(s1, s2)
        model.add_edge(s1, s3)
        model.add_edge(s2, s3)
        model.bake()
        self.model = model

        meta = []
        for i in range(self.model.node_count()):
            meta.append({
                "name": None,
                "type": "Categorical",
                "size": 2,
                "i2s": ['T', 'F']
        })
        self.meta = meta


class GeneralMaker(MultivariateMaker):
    def __init__(self):
        Pollution = DiscreteDistribution({'F': 0.9, 'T': 0.1})
        Smoker = DiscreteDistribution({'T': 0.3, 'F': 0.7})
        Cancer = ConditionalProbabilityTable(
            [['T','T','T',0.05],
            ['T','T','F',0.95],
            ['T','F','T',0.02],
            ['T','F','F',0.98],
            ['F','T','T',0.03],
            ['F','T','F',0.97],
            ['F','F','T',0.001],
            ['F','F','F',0.999],
            ],[Pollution,Smoker])
        XRay = ConditionalProbabilityTable(
            [['T','T',0.9],
            ['T','F',0.1],
            ['F','T',0.2],
            ['F','F',0.8],
            ],[Cancer])
        Dyspnoea = ConditionalProbabilityTable(
            [['T','T',0.65],
            ['T','F',0.35],
            ['F','T',0.3],
            ['F','F',0.7],
            ],[Cancer])
        s1 = Node(Pollution, name="Pollution")
        s2 = Node(Smoker, name="Smoker")
        s3 = Node(Cancer, name="Cancer")
        s4 = Node(XRay, name="XRay")
        s5 = Node(Dyspnoea, name="Dyspnoea")

        model = BayesianNetwork("Lung Cancer")
        model.add_states(s1, s2, s3, s4, s5)
        model.add_edge(s1, s3)
        model.add_edge(s2, s3)
        model.add_edge(s3, s4)
        model.add_edge(s3, s5)
        model.bake()
        self.model = model

        meta = []
        name_mapper = ["Pollution", "Smoker", "Cancer", "XRay", "Dyspnoea"]
        for i in range(self.model.node_count()):
            meta.append({
                "name": name_mapper[i],
                "type": "Categorical",
                "size": 2,
                "i2s": ['T', 'F']
        })
        self.meta = meta


if __name__ == "__main__":
    supported_distributions = {'chain': ChainMaker, 'tree': TreeMaker,'fc':FCMaker,'general': GeneralMaker}

    parser = argparse.ArgumentParser(description='Generate simulated Data for a distribution')
    parser.add_argument('distribution', type = str, help = 'specify type of distributions to sample from')
    parser.add_argument('--sample', type=int, default=20000,
                    help='maximum samples in the simulated data.')

    args = parser.parse_args()
    dist = args.distribution
    num_sample = args.sample
    if dist in supported_distributions:
        maker = supported_distributions[dist]()
        samples = maker.sample(num_sample)

    output_dir = "data/simulated"
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            pass
    # Store simulated data
    with open("{}/{}.json".format(output_dir, dist), 'w') as f:
        json.dump(maker.meta, f, sort_keys=True, indent=4, separators=(',', ': '))
    with open("{}/{}_structure.json".format(output_dir, dist), 'w') as f:
        f.write(maker.model.to_json())
    np.savez("{}/{}.npz".format(output_dir, dist), train=samples[:len(samples)//2], test=samples[len(samples)//2:])
