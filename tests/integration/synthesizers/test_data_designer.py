"""Integration tests for the DataDesignerSynthesizer."""

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from sdgym.synthesizers import DataDesignerSynthesizer

pytest.importorskip('data_designer.interface')


@pytest.fixture
def data():
    n_rows = 50
    rng = np.random.default_rng(0)
    amenities_fee = rng.uniform(0, 40, size=n_rows).round(2)
    amenities_fee[rng.random(n_rows) < 0.2] = np.nan
    checkin = pd.Timestamp('2020-01-03') + pd.to_timedelta(rng.integers(0, 360, n_rows), unit='D')
    room_type = rng.choice(['BASIC', 'DELUXE', 'SUITE', None], size=n_rows, p=[0.5, 0.2, 0.1, 0.2])
    return pd.DataFrame({
        'guest_id': [f'G{index:04d}' for index in range(n_rows)],
        'has_rewards': rng.choice([True, False], size=n_rows, p=[0.3, 0.7]),
        'room_type': room_type,
        'num_guests': rng.integers(1, 5, size=n_rows),
        'amenities_fee': amenities_fee,
        'checkin_date': checkin.strftime('%d %b %Y'),
    })


@pytest.fixture
def metadata_dict():
    return {
        'tables': {
            'guests': {
                'primary_key': 'guest_id',
                'columns': {
                    'guest_id': {'sdtype': 'id', 'regex_format': 'G[0-9]{4}'},
                    'has_rewards': {'sdtype': 'boolean'},
                    'room_type': {'sdtype': 'categorical'},
                    'num_guests': {'sdtype': 'numerical'},
                    'amenities_fee': {'sdtype': 'numerical'},
                    'checkin_date': {'sdtype': 'datetime', 'datetime_format': '%d %b %Y'},
                },
            }
        }
    }


def test_data_designer_end_to_end(data, metadata_dict, tmp_path, monkeypatch):
    """Test fitting and sampling without metrics.

    Only sampler based columns are used, so no model provider or API key is needed.
    """
    # Setup
    monkeypatch.chdir(tmp_path)
    synthesizer = DataDesignerSynthesizer()
    n_samples = 30

    # Run
    trained_synthesizer = synthesizer.get_trained_synthesizer(data, metadata_dict)
    trained_synthesizer = cloudpickle.loads(cloudpickle.dumps(trained_synthesizer))
    sampled_data = synthesizer.sample_from_synthesizer(trained_synthesizer, n_samples=n_samples)

    # Assert
    assert not list(tmp_path.iterdir()), 'No artifacts should be left in the working directory'
    assert isinstance(sampled_data, pd.DataFrame)
    assert sampled_data.shape == (n_samples, data.shape[1])
    assert set(sampled_data.columns) == set(data.columns)

    assert sampled_data['guest_id'].is_unique
    room_types = sampled_data['room_type']
    assert set(room_types.dropna()) <= {'BASIC', 'DELUXE', 'SUITE'}
    assert room_types.isna().any(), 'Missing values should be sampled as a category'
    assert not (room_types == '__sdgym_null__').any()
    assert set(sampled_data['has_rewards'].astype(int)) <= {0, 1}

    num_guests = sampled_data['num_guests'].astype(float)
    assert num_guests.between(data['num_guests'].min(), data['num_guests'].max()).all()
    assert (num_guests % 1 == 0).all()

    amenities_fee = sampled_data['amenities_fee'].astype(float)
    assert amenities_fee.between(data['amenities_fee'].min(), data['amenities_fee'].max()).all()

    real_dates = pd.to_datetime(data['checkin_date'], format='%d %b %Y')
    sampled_dates = pd.to_datetime(sampled_data['checkin_date'])
    assert sampled_dates.between(real_dates.min(), real_dates.max()).all()


def test_data_designer_with_metadata_v2_ranges(data, metadata_dict, tmp_path, monkeypatch):
    """Test the Metadata V2 range hints bound the sampled data."""
    # Setup
    monkeypatch.chdir(tmp_path)
    columns = metadata_dict['tables']['guests']['columns']
    columns['room_type']['range_values'] = ['BASIC', 'DELUXE', 'SUITE', 'PENTHOUSE']
    columns['num_guests'].update({'range_min': 2, 'range_max': 3, 'decimal_places': 0})
    columns['amenities_fee'].update({'range_min': 100.0, 'range_max': 200.0, 'decimal_places': 1})
    columns['checkin_date'].update({'range_min': '01 Jan 2022', 'range_max': '31 Jan 2022'})
    synthesizer = DataDesignerSynthesizer()

    # Run
    trained_synthesizer = synthesizer.get_trained_synthesizer(data, metadata_dict)
    sampled_data = synthesizer.sample_from_synthesizer(trained_synthesizer, n_samples=30)

    # Assert
    assert len(sampled_data) == 30
    assert 'PENTHOUSE' not in set(sampled_data['room_type'])
    assert sampled_data['num_guests'].astype(float).between(2, 3).all()
    assert sampled_data['amenities_fee'].astype(float).between(100.0, 200.0).all()
    sampled_dates = pd.to_datetime(sampled_data['checkin_date'])
    assert sampled_dates.between(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-31')).all()


@pytest.mark.parametrize('cleanup_artifacts', [True, False])
def test_data_designer_artifact_path(data, metadata_dict, tmp_path, monkeypatch, cleanup_artifacts):
    """Test artifacts are written to ``artifact_path`` and cleaned up only when requested."""
    # Setup
    working_directory = tmp_path / 'cwd'
    artifact_path = tmp_path / 'artifacts'
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)
    model_kwargs = {'artifact_path': artifact_path, 'cleanup_artifacts': cleanup_artifacts}
    monkeypatch.setattr(DataDesignerSynthesizer, '_MODEL_KWARGS', model_kwargs)
    synthesizer = DataDesignerSynthesizer()

    # Run
    trained_synthesizer = synthesizer.get_trained_synthesizer(data, metadata_dict)
    first_sample = synthesizer.sample_from_synthesizer(trained_synthesizer, n_samples=10)
    second_sample = synthesizer.sample_from_synthesizer(trained_synthesizer, n_samples=20)

    # Assert
    assert len(first_sample) == 10
    assert len(second_sample) == 20
    assert first_sample['room_type'].dropna().isin(data['room_type'].dropna().unique()).all()
    assert not list(working_directory.iterdir())

    datasets = list(artifact_path.iterdir())
    if cleanup_artifacts:
        assert datasets == []
    else:
        assert len(datasets) == 2
        assert all(list(dataset.rglob('*.parquet')) for dataset in datasets)
