import cloudpickle
import pandas as pd
from sdgym.benchmark import benchmark_single_table
from sdgym.synthesizers.generate import create_single_table_synthesizer
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer

def get_trained_synth(data, metadata):
    metadata = SingleTableMetadata.load_from_dict(metadata)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    return synthesizer

def get_trained_synth_v2(data, metadata):
    metadata = SingleTableMetadata.load_from_dict(metadata)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    return synthesizer

def sample_synth(trained_synthesizer, num_samples):
    return trained_synthesizer.sample(num_samples)

custom_synthesizer = create_single_table_synthesizer('SimpleGaussianCopula', get_trained_synth, sample_synth)
custom_synthesizer_v2 = create_single_table_synthesizer('SimpleGaussianCopulaV2', get_trained_synth_v2, sample_synth)

if __name__ == '__main__':
    output = benchmark_single_table(
        synthesizers=[],
        sdv_datasets=['fake_hotel_guests'],
        timeout=120,
        sdmetrics=[],
        custom_synthesizers=[custom_synthesizer, custom_synthesizer_v2],
    )
    with pd.option_context('display.max_columns', None):
        print(output)
