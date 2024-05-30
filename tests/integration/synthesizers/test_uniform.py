"""Module to test the UniformSynthesizer."""

import numpy as np
import pandas as pd

from sdgym.synthesizers.uniform import UniformSynthesizer


def test_uniform_synthesizer():
    """Ensure the produced samples are indeed uniform."""
    # Setup
    n_samples = 10000
    num_values = np.random.normal(size=n_samples)
    num_values[np.random.random(size=n_samples) < 0.1] = np.nan

    cat_values = np.random.choice(['a', 'b', 'c', np.nan], size=n_samples, p=[0.1, 0.2, 0.6, 0.1])
    bool_values = np.random.choice([True, False], size=n_samples, p=[0.3, 0.7])

    dates = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', np.nan])
    date_values = np.random.choice(dates, size=n_samples, p=[0.1, 0.2, 0.6, 0.1])

    data = pd.DataFrame({
        'num': num_values,
        'cat': cat_values,
        'bool': bool_values,
        'date': date_values,
    })

    uniform_synthesizer = UniformSynthesizer()

    # Run
    trained_synthesizer = uniform_synthesizer.get_trained_synthesizer(data, {})
    samples = uniform_synthesizer.sample_from_synthesizer(trained_synthesizer, n_samples)

    # Assert numerical values are uniform
    min_val = samples['num'].min()
    max_val = samples['num'].max()
    interval = (max_val - min_val) / 3

    n_values_interval1 = sum(samples['num'].between(min_val, min_val + interval))
    n_values_interval2 = sum(samples['num'].between(min_val + interval, max_val - interval))
    n_values_interval3 = sum(samples['num'].between(max_val - interval, max_val))

    assert n_values_interval2 * 0.9 < n_values_interval1 < n_values_interval2 * 1.1
    assert n_values_interval3 * 0.9 < n_values_interval1 < n_values_interval3 * 1.1

    # Assert categorical values are uniform
    a_values = sum(samples['cat'] == 'a')
    b_values = sum(samples['cat'] == 'b')
    c_values = sum(samples['cat'] == 'c')

    assert b_values * 0.9 < a_values < b_values * 1.1
    assert c_values * 0.9 < a_values < c_values * 1.1

    # Assert boolean values are uniform
    a_values = sum(samples['bool'] == True)  # noqa: E712
    b_values = sum(samples['bool'] == False)  # noqa: E712

    assert b_values * 0.9 < a_values < b_values * 1.1

    # Assert datetime values are uniform
    min_val = samples['date'].min()
    max_val = samples['date'].max()
    interval = (max_val - min_val) / 3

    n_values_interval1 = sum(samples['date'].between(min_val, min_val + interval))
    n_values_interval2 = sum(samples['date'].between(min_val + interval, max_val - interval))
    n_values_interval3 = sum(samples['date'].between(max_val - interval, max_val))

    assert n_values_interval2 * 0.9 < n_values_interval1 < n_values_interval2 * 1.1
    assert n_values_interval3 * 0.9 < n_values_interval1 < n_values_interval3 * 1.1
