import argparse
import json
import os
import pandas as pd
import numpy as np
import re

from pomegranate import DiscreteDistribution, ConditionalProbabilityTable, Node, BayesianNetwork
from .. import utils

def map_col(index2str, values):
    mapper = dict([(k, v) for v, k in enumerate(index2str)])
    return [mapper[item.decode('utf8')] for item in values]

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

        data = np.empty((n, len(nodes_parents)), dtype='S128')
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
                        tmp = data[_id, parents_map[i]]
                        try:
                            tmp = tmp.decode('utf8')
                        except:
                            pass
                        values[parents[i]] = tmp
                    data[_id, current] = distribution.sample(parent_values=values)

        data_t = np.zeros(data.shape)
        for col_id in range(data.shape[1]):
            data_t[:, col_id] = map_col(self.meta[col_id]['i2s'], data[:, col_id])
        return data_t




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
                "name": chr(ord('A') + i),
                "type": "categorical",
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
                "name": chr(ord('A') + i),
                "type": "categorical",
                "size": 3,
                "i2s": ['1', '2', '3']
        })
        meta.append({
                "name": "C",
                "type": "categorical",
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
                "type": "categorical",
                "size": 2,
                "i2s": ['T', 'F']
        })
        meta[0]['name'] = 'Rain'
        meta[1]['name'] = 'Sprinkler'
        meta[2]['name'] = 'Wet'
        self.meta = meta


class GeneralMaker(MultivariateMaker):
    def __init__(self):
        Pollution = DiscreteDistribution({'F': 0.9, 'T': 0.1})
        Smoker = DiscreteDistribution({'T': 0.3, 'F': 0.7})
        print(Smoker)
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
        print(Cancer)
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
                "type": "categorical",
                "size": 2,
                "i2s": ['T', 'F']
        })
        self.meta = meta

class BIFMaker(MultivariateMaker):
    def __init__(self, filename):
        with open(filename) as f:
            bif = f.read()
        vars = re.findall(r"variable[^\{]+{[^\}]+}", bif)
        probs = re.findall(r"probability[^\{]+{[^\}]+}", bif)

        var_nodes = {}
        var_index_to_name = []
        edges = []

        self.meta = []
        todo = set()
        for v, p in zip(vars, probs):
            m = re.search(r"variable\s+([^\{\s]+)\s+", v)
            v_name = m.group(1)
            m = re.search(r"type\s+discrete\s+\[\s*(\d+)\s*\]\s*\{([^\}]+)\}", v)
            v_opts_n = int(m.group(1))
            v_opts = m.group(2).replace(',', ' ').split()

            assert v_opts_n == len(v_opts)
            # print(v_name, v_opts_n, v_opts)

            m = re.search(r"probability\s*\(([^)]+)\)", p)
            cond = m.group(1).replace('|', ' ').replace(',', ' ').split()
            assert cond[0] == v_name
            # print(cond)

            self.meta.append({
                "name": v_name,
                "type": "categorical",
                "size": v_opts_n,
                "i2s": v_opts
            })
            if len(cond) == 1:
                m = re.search(r"table([e\-\d\.\s,]*);", p)
                margin_p = m.group(1).replace(',', ' ').split()
                margin_p = [float(x) for x in margin_p]
                assert abs(sum(margin_p) - 1) < 1e-6
                assert len(margin_p) == v_opts_n
                margin_p = dict(zip(v_opts, margin_p))

                var_index_to_name.append(v_name)
                tmp = DiscreteDistribution(margin_p)
                # print(tmp)
                var_nodes[v_name] = tmp
            else:
                m_iter = re.finditer(r"\(([^)]*)\)([\s\d\.,\-e]+);", p)
                cond_p_table = []
                for m in m_iter:
                    cond_values = m.group(1).replace(',', ' ').split()
                    cond_p = m.group(2).replace(',', ' ').split()
                    cond_p = [float(x) for x in cond_p]
                    assert len(cond_values) == len(cond) - 1
                    assert len(cond_p) == v_opts_n
                    assert abs(sum(cond_p) - 1) < 1e-6

                    for opt, opt_p in zip(v_opts, cond_p):
                        cond_p_table.append(cond_values + [opt, opt_p])
                var_index_to_name.append(v_name)

                tmp = (cond_p_table, cond)
                # print(tmp)
                var_nodes[v_name] = tmp
                for x in cond[1:]:
                    edges.append((x, v_name))
                todo.add(v_name)

        while len(todo) > 0:
            # print(todo)
            for v_name in todo:
                # print(v_name, type(var_nodes[v_name]))
                cond_p_table, cond = var_nodes[v_name]
                flag = True
                for y in cond[1:]:
                    if y in todo:
                        flag = False
                        break
                if flag:
                    cond_t = [var_nodes[x] for x in cond[1:]]
                    var_nodes[v_name] = ConditionalProbabilityTable(cond_p_table, cond_t)
                    todo.remove(v_name)
                    break

        for x in var_index_to_name:
            var_nodes[x] = Node(var_nodes[x], name=x)

        var_nodes_list = [var_nodes[x] for x in var_index_to_name]
        # print(var_nodes_list)
        model = BayesianNetwork("tmp")
        model.add_states(*var_nodes_list)

        for edge in edges:
            model.add_edge(var_nodes[edge[0]], var_nodes[edge[1]])
        model.bake()
        # print(model.to_json())
        self.model = model


if __name__ == "__main__":
    supported_distributions = {'chain': ChainMaker, 'tree': TreeMaker,'fc':FCMaker,'general': GeneralMaker}

    parser = argparse.ArgumentParser(description='Generate simulated Data for a distribution')
    parser.add_argument('distribution', type = str, help = 'specify type of distributions to sample from')
    parser.add_argument('--sample', type=int, default=10000,
                    help='maximum samples in the simulated data.')

    args = parser.parse_args()
    dist = args.distribution
    num_sample = args.sample * 2
    if dist in supported_distributions:
        maker = supported_distributions[dist]()
        samples = maker.sample(num_sample)
    else:
        biffile = "data/raw/bif/" + dist + ".bif"
        if os.path.exists(biffile):
            maker = BIFMaker(biffile)
            samples = maker.sample(num_sample)
        else:
            assert 0

    # assert 0

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

    utils.verify("{}/{}.npz".format(output_dir, dist),
        "{}/{}.json".format(output_dir, dist))
