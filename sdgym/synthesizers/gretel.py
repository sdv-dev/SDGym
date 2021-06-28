import tempfile

import numpy as np

from sdgym.synthesizers.base import SingleTableBaseline

try:
    from gretel_synthetics.batch import DataFrameBatch
except ImportError:
    DataFrameBatch = None


class Gretel(SingleTableBaseline):
    """Class to represent Gretel's neural network model."""

    def __init__(self, max_lines=0, max_line_len=2048, epochs=None, vocab_size=20000,
                 gen_lines=None, dp=False, field_delimiter=",", overwrite=True,
                 checkpoint_dir=None):
        if DataFrameBatch is None:
            raise ImportError('Please install gretel-synthetics using `pip install sdgym[gretel]`')

        self.max_lines = max_lines
        self.max_line_len = max_line_len
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.gen_lines = gen_lines
        self.dp = dp
        self.field_delimiter = field_delimiter
        self.overwrite = overwrite
        self.checkpoint_dir = checkpoint_dir or tempfile.TemporaryDirectory().name

    def _fit_sample(self, data, metadata):
        config = {
            'max_lines': self.max_lines,
            'max_line_len': self.max_line_len,
            'epochs': self.epochs or data.shape[1] * 3,  # value recommended by Gretel
            'vocab_size': self.vocab_size,
            'gen_lines': self.gen_lines or data.shape[0],
            'dp': self.dp,
            'field_delimiter': self.field_delimiter,
            'overwrite': self.overwrite,
            'checkpoint_dir': self.checkpoint_dir
        }
        batcher = DataFrameBatch(df=data, config=config)
        batcher.create_training_data()
        batcher.train_all_batches()
        batcher.generate_all_batch_lines()
        synth_data = batcher.batches_to_df()
        return synth_data


class PreprocessedGretel(Gretel):
    """Class that uses RDT to make all columns numeric before using Gretel's model."""

    CONVERT_TO_NUMERIC = True

    @staticmethod
    def make_numeric(val):
        if type(val) in [float, int]:
            return val

        if isinstance(val, str) and val.isnumeric():
            return float(val)

        return np.nan

    def _fix_numeric_columns(self, data, metadata):
        fields_metadata = metadata['fields']
        for field in data:
            if field in fields_metadata and fields_metadata.get(field).get('type') == 'id':
                continue

            data[field] = data[field].apply(self.make_numeric)
            avg = data[field].mean() if not np.isnan(data[field].mean()) else 0
            data[field] = data[field].fillna(round(avg))

    def _fit_sample(self, data, metadata):
        synth_data = super()._fit_sample(data, metadata)
        self._fix_numeric_columns(synth_data, metadata)
        return synth_data
