import pathlib

import pandas as pd
import tqdm


def collect_results(input_path, output_file):
    """Collect the results in the given input directory, and
    write all the results into one csv file.

    Args:
        input_path (str):
            The path of the directory that the results files
            will be read from.
        output_file (str):
            If ``output_file`` is provided, the consolidated
            results will be written there. Otherwise, they
            will be written to ``input_path``/results.csv.
    """
    print(f'Reading results from {input_path}')
    run_path = pathlib.Path(input_path)

    scores = []
    for path in tqdm.tqdm(list(run_path.glob('**/*.csv'))):
        scores.append(pd.read_csv(path))

    scores = pd.concat(scores).drop_duplicates()

    if output_file:
        output = output_file
    else:
        output = run_path / 'results.csv'

    print(f'Storing results at {output}')
    scores.to_csv(output, index=False)
