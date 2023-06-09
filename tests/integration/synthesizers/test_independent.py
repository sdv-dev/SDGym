"""Module to test the IndependentSynthesizer."""
import numpy as np
import pandas as pd

from sdgym.synthesizers.independent import IndependentSynthesizer


def test_independent_synthesizer():
    """Ensure all sdtypes can be sampled."""
    # Setup
    n_samples = 10000
    num_values = np.random.normal(size=n_samples)
    cat_values = np.random.choice(['a', 'b', 'c'], size=n_samples, p=[.1, .2, .7])
    bool_values = np.random.choice([True, False], size=n_samples, p=[.3, .7])

    dates = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
    date_values = np.random.choice(dates, size=n_samples, p=[.1, .2, .7])

    data = pd.DataFrame({
        'num': num_values,
        'cat': cat_values,
        'bool': bool_values,
        'date': date_values,
    })

    independent_synthesizer = IndependentSynthesizer()

    # Run
    trained_synthesizer = independent_synthesizer.get_trained_synthesizer(data, {})
    samples = independent_synthesizer.sample_from_synthesizer(trained_synthesizer, n_samples)

    # Assert
    assert samples['num'].between(-10, 10).all()
    assert ((samples['cat'] == 'a') | (samples['cat'] == 'b') | (samples['cat'] == 'c')).all()
    assert ((samples['bool'] == True) | (samples['bool'] == False)).all()  # noqa: E712
    assert samples['date'].between(
        pd.to_datetime('2019-01-01'),
        pd.to_datetime('2021-01-01')
    ).all()
