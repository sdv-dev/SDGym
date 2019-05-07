import argparse
import logging
import glob
import os
import json
import numpy as np


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Launch one synthesizer on one or multiple datasets.')
parser.add_argument('datasets', type=str, nargs='*',
                    help='a list of datasets, empty means all datasets.')
parser.add_argument('--repeat', type=int, default=3,
                    help='run generater multiple times.')

parser.add_argument('--output', type=str, default='output',
                    help='output dir')
parser.add_argument('--name', type=str, default='',
                    help='model name, default is model class name.')

parser.add_argument('--sample', type=int, default=50000,
                    help='maximum samples in the synthetic data.')

class SynthesizerBase(object):
    """docstring for Synthesizer."""

    supported_datasets = [
        'asia', 'alarm', 'child', 'insurance', 'grid', 'gridr', 'ring',
        'adult', 'credit', 'census',
        'news', 'covtype', 'intrusion', 'mnist12', 'mnist28']

    def train(self, train_data):
        pass

    def generate(self, n):
        pass

    def init(self, meta, working_dir):
        pass


def run(synthesizer):
    assert isinstance(synthesizer, SynthesizerBase)

    args = parser.parse_args()
    datasets = args.datasets
    name = args.name
    if name == "":
        name = synthesizer.__class__.__name__

    output = "{}/{}".format(args.output, name)

    if not os.path.exists(output):
        os.makedirs(output)
    logging.info("Use output dir {}".format(output))

    repeat = args.repeat

    if datasets == []:
        datasets = synthesizer.supported_datasets
    else:
        for item in datasets:
            if not item in synthesizer.supported_datasets:
                logging.warning("Dataset {} is not supported by {}.".format(item, name))



    for dataset in datasets:

        data_filename = glob.glob("data/*/{}.npz".format(dataset))
        meta_filename = glob.glob("data/*/{}.json".format(dataset))

        if len(data_filename) != 1:
            logging.warning("Skip. Can't find dataset {}. ".format(dataset))
            continue

        if len(meta_filename) != 1:
            logging.warning("Skip. Can't find meta {}. ".format(dataset))
            continue

        data = np.load(data_filename[0])['train']
        with open(meta_filename[0]) as f:
            meta = json.load(f)


        for i in range(repeat):
            if glob.glob("{}/{}_{}_*.npz".format(output, dataset, i)):
                logging.warning("Skip. {} results on {}_{} exists.".format(name, dataset, i))
                continue

            logging.info("Generating {} iter {}".format(dataset, i))
            working_dir = "{}/ckpt_{}_{}".format(output, dataset, i)
            synthesizer.init(meta, working_dir)
            synthesizer.train(data)
            sample = min(args.sample, data.shape[0])
            generated = synthesizer.generate(sample)

            if len(generated) == 0:
                logging.warning("{} fails on {}. ".format(name, dataset))

            for step, syn in generated:
                np.savez("{}/{}_{}_{}.npz".format(output, dataset, i, step), syn=syn)
