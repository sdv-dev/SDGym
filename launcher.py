import subprocess
import argparse
import os
import logging
import glob
import shutil

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='A launcher for easily launch experiments.')

parser.add_argument('--force', dest='force',
                    action='store_true', help='overwrite results.')
parser.set_defaults(force=False)

parser.add_argument('--datasets', type=str, nargs='*', default=[],
                    help='a list of datasets, empty means all datasets.')

parser.add_argument('--repeat', type=int, default=3,
                    help='a list of datasets, empty means all datasets.')

parser.add_argument('synthesizer', type=str,
                    help='select a data synthesizer, e.g. identity')

def case_insensitive(x):
    t = ""
    for c in x:
        t += "[{}{}]".format(c.upper(), c.lower())
    return t

if __name__ == "__main__":
    args = parser.parse_args()

    if args.force:
        pattern = case_insensitive(args.synthesizer)
        output_folder = glob.glob("output/{}Synthesizer".format(pattern))
        if len(output_folder) != 0:
            output_folder = output_folder[0]
            shutil.rmtree(output_folder)
            logging.warning("remove existing results {}".format(output_folder))

    subprocess.call(["python3", "-m",
        "synthetic_data_benchmark.synthesizer.{}_synthesizer".format(args.synthesizer.lower()),
        "--repeat", str(args.repeat)] + args.datasets)

    pattern = case_insensitive(args.synthesizer)
    output_folder = glob.glob("output/{}Synthesizer".format(pattern))
    if args.force:
        subprocess.call(["python3", "-m", "synthetic_data_benchmark.evaluator.evaluate", output_folder[0], "--force"])
    else:
        subprocess.call(["python3", "-m", "synthetic_data_benchmark.evaluator.evaluate", output_folder[0]])

    subprocess.call(["python3", "-m", "synthetic_data_benchmark.evaluator.summary"])
