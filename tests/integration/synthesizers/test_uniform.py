import numpy as np
import pandas as pd
from sdv.metadata.single_table import SingleTableMetadata

from sdgym.synthesizers.uniform import UniformSynthesizer


def test_uniform_synthesizer():
    """Ensure the produced samples are indeed uniform."""
    # Setup
    n_samples = 10000
    data = pd.DataFrame({
        'num': np.random.normal(size=n_samples),
        'cat': np.random.choice(['a', 'b', 'c'], size=n_samples, p=[.1, .2, .7])
    })

    metadata = SingleTableMetadata()
    metadata.add_column('num', sdtype='numerical')
    metadata.add_column('cat', sdtype='categorical')
    uniform_synthesizer = UniformSynthesizer()

    # Run
    trained_synthesizer = uniform_synthesizer.get_trained_synthesizer(data, metadata.to_dict())
    samples = uniform_synthesizer.sample_from_synthesizer(trained_synthesizer, n_samples)

    # Assert numerical values are uniform
    min_val = samples['num'].min()
    max_val = samples['num'].max()
    interval = (max_val - min_val) / 3

    n_values_interval1 = sum(samples['num'].between(min_val, min_val + interval))
    n_values_interval2 = sum(samples['num'].between(min_val + interval, max_val - interval))
    n_values_interval3 = sum(samples['num'].between(max_val - interval, max_val))

    assert n_values_interval2 * .9 < n_values_interval1 < n_values_interval2 * 1.1
    assert n_values_interval3 * .9 < n_values_interval1 < n_values_interval3 * 1.1

    # Assert categorical values are uniform
    a_values = sum(samples['cat'] == 'a')
    b_values = sum(samples['cat'] == 'b')
    c_values = sum(samples['cat'] == 'c')

    assert b_values * .9 < a_values < b_values * 1.1
    assert c_values * .9 < a_values < c_values * 1.1
