import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd
import tabulate

GM_TITLE = 'Gaussian Mixture Simulated Data'
BN_TITLE = 'Bayesian Networks Simulated Data'
RW_TITLE = 'Real World Datasets'

GM_COLUMNS = ['grid/syn_likelihood', 'grid/test_likelihood', 'gridr/syn_likelihood',
              'gridr/test_likelihood', 'ring/syn_likelihood', 'ring/test_likelihood']
BN_COLUMNS = ['asia/syn_likelihood', 'asia/test_likelihood', 'alarm/syn_likelihood',
              'alarm/test_likelihood', 'child/syn_likelihood',
              'child/test_likelihood', 'insurance/syn_likelihood',
              'insurance/test_likelihood']
RW_COLUMNS = ['adult/f1', 'census/f1', 'credit/f1', 'covtype/macro_f1',
              'intrusion/macro_f1', 'mnist12/accuracy', 'mnist28/accuracy',
              'news/r2']

DROP_SYNTHESIZERS = ['IdentitySynthesizer', 'IndependentSynthesizer', 'UniformSynthesizer']

BASE_DIR = os.path.dirname(__file__)
LEADERBOARD_PATH = os.path.join(BASE_DIR, 'leaderboard.csv')


def load_results(files):
    results = dict()
    for filename in files:
        version = os.path.basename(filename).replace('.csv', '')
        version_results = pd.read_csv(filename, index_col=0)
        version_results.index.name = 'Synthesizer'
        for synthesizer in DROP_SYNTHESIZERS:
            try:
                version_results.drop(synthesizer, inplace=True)
            except KeyError:
                pass   # already not there

        results[version] = {
            GM_TITLE: version_results[GM_COLUMNS],
            BN_TITLE: version_results[BN_COLUMNS],
            RW_TITLE: version_results[RW_COLUMNS],
        }

    return results


def get_wins(scores):
    is_winner = scores.rank(method='min', ascending=False) == 1
    return is_winner.sum(axis=1)


def get_summary(results, summary_function):
    summary = defaultdict(dict)
    for version, scores in results.items():
        for section in [GM_TITLE, BN_TITLE, RW_TITLE]:
            summary[section][version] = summary_function(scores[section])

    for section in [GM_TITLE, BN_TITLE, RW_TITLE]:
        section_df = pd.DataFrame(summary[section])
        section_df.index.name = 'Synthesizer'
        columns = section_df.columns.sort_values(ascending=False)
        summary[section] = section_df[columns].fillna('N/E')

    return summary


def add_sheet(dfs, name, writer, cell_fmt, index_fmt, header_fmt):
    startrow = 0
    widths = [0]
    if not isinstance(dfs, dict):
        dfs = {None: dfs}

    for df_name, df in dfs.items():
        df = df.fillna('N/E').sort_index().reset_index()
        startrow += bool(df_name)
        df.to_excel(writer, sheet_name=name, startrow=startrow + 1, index=False, header=False)

        worksheet = writer.sheets[name]

        if df_name:
            worksheet.write(startrow - 1, 0, df_name, index_fmt)
            widths[0] = max(widths[0], len(df_name))

        for idx, column in enumerate(df.columns):
            worksheet.write(startrow, idx, column, header_fmt)
            width = max(len(column), *df[column].astype(str).str.len()) + 1
            if len(widths) > idx:
                widths[idx] = max(widths[idx], width)
            else:
                widths.append(width)

        startrow += len(df) + 2

    for idx, width in enumerate(widths):
        fmt = cell_fmt if idx else index_fmt
        worksheet.set_column(idx, idx, width + 1, fmt)


def write_results(results, summary, output):
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    cell_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10"
    })
    index_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10",
        "bold": True,
    })
    header_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10",
        "bold": True,
        "bottom": 1
    })

    add_sheet(summary, 'Number of wins per version', writer, cell_fmt, index_fmt, header_fmt)

    for version in reversed(sorted(results.keys())):
        add_sheet(results[version], version, writer, cell_fmt, index_fmt, header_fmt)

    writer.save()


def summarize_results(input_paths, output_path):
    results = load_results(input_paths)
    summary = get_summary(results, get_wins)
    write_results(results, summary, output_path)

    return summary


def _dataset_summary(grouped_df):
    dataset = grouped_df.name[1]
    scores = grouped_df.mean().dropna()
    scores.index = dataset + '/' + scores.index

    return scores


def make_leaderboard(scores, add_leaderboard=True, leaderboard_path=None,
                     replace_existing=True, output_path=None):
    """Make a leaderboard out of individual synthesizer scores.

    If ``add_leaderboard`` is ``True``, append the obtained scores to the leaderboard
    stored in the ``lederboard_path``. By default, the leaderboard used is the one which
    is included in the package, which contains the scores obtained by the SDGym Synthesizers.

    If ``replace_existing`` is ``True`` and any of the given synthesizers already existed
    in the leaderboard, the old rows are dropped.

    Args:
        scores (list):
            List of DataFrames with scores or paths to CSV files.
        add_leaderboard (bool):
            Whether to append the obtained scores to the previous leaderboard or not. Defaults
            to ``True``.
        leaderboard_path (str):
            Path to where the leaderboard is stored. Defaults to the leaderboard included
            with the package, which contains the scores obtained by the SDGym synthesizers.
        replace_existing (bool):
            Whether to replace old scores or keep them in the returned leaderboard. Defaults
            to ``True``.
        output_path (str):
            If an ``output_path`` is given, the generated leaderboard will be stored in the
            indicated path as a CSV file. The given path must be a complete path including
            the ``.csv`` filename.

    Returns:
        pandas.DataFrame or None:
            If not ``output_path`` is given, a table containing one row per synthesizer and
            one column for each dataset and metric is returned. Otherwise, there is no output.
    """
    if isinstance(scores, str) and os.path.isdir(scores):
        scores = [
            pd.read_csv(os.path.join(scores, path), index_col=0)
            for path in os.listdir(scores)
        ]

    scores = pd.concat(scores, ignore_index=True)
    scores = scores.drop(['iteration', 'name'], axis=1, errors='ignore')

    leaderboard = scores.groupby(['synthesizer', 'dataset']).apply(_dataset_summary)
    if isinstance(leaderboard, pd.Series):
        leaderboard = leaderboard.droplevel(-2).unstack(-1)
    else:
        leaderboard = leaderboard.droplevel(-1)

    leaderboard['timestamp'] = datetime.utcnow()

    if add_leaderboard:
        old_leaderboard = pd.read_csv(
            leaderboard_path or LEADERBOARD_PATH,
            index_col=0,
            parse_dates=['timestamp']
        ).reindex(columns=leaderboard.columns)
        if replace_existing:
            old_leaderboard.drop(labels=leaderboard.index, errors='ignore', inplace=True)

        leaderboard = old_leaderboard.append(leaderboard, sort=False)

    if output_path:
        os.makedirs(os.path.dirname(os.path.realpath(output_path)), exist_ok=True)
        leaderboard.to_csv(output_path)
    else:
        return leaderboard


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Results Summary')
    parser.add_argument('input', nargs='+', help='Input path with results.')
    parser.add_argument('output', help='Output file.')

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    summary = summarize_results(args.input, args.output)

    for title, section in summary.items():
        print('\n### {}\n'.format(title))
        print(tabulate.tabulate(
            section.reset_index(),
            tablefmt='github',
            headers=['Synthesizer'] + list(section.columns),
            showindex=False
        ))
