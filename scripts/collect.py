import argparse
import pathlib
import sys

import pandas as pd


def collect():
    parser = argparse.ArgumentParser(description='Collect SDGym Results')
    parser.add_argument('input', help='Run folder path.')
    parser.add_argument('output', help='Output file.')

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    run_path = pathlib.Path(args.input)

    scores = []
    for path in run_path.glob('**/*.csv'):
        scores.append(pd.read_csv(path))

    scores = pd.concat(scores).drop_duplicates()

    scores.to_csv(args.output, index=False)


if __name__ == '__main__':
    collect()
