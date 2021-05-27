from sdgym.s3 import read_csv_from_path, write_csv


def collect_results(input_path, output_file=None, aws_key=None, aws_secret=None):
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
        aws_key (str):
            If an ``aws_key`` is provided, the given access
            key id will be used to read from and/or write to
            any s3 paths.
        aws_secret (str):
            If an ``aws_secret`` is provided, the given secret
            access key will be used to read from and/or write to
            any s3 paths.
    """
    print(f'Reading results from {input_path}')
    scores = read_csv_from_path(input_path, aws_key, aws_secret)
    scores = scores.drop_duplicates()

    if output_file:
        output = output_file
    else:
        output = f'{input_path}/results.csv'

    print(f'Storing results at {output}')
    write_csv(scores, output, aws_key, aws_secret)
