from sdgym import load_dataset
from sdgym.synthesizers import ClavaDDPMSynthesizer


def test_clavaddpm_end_to_end():
    """Test it without metrics."""
    # Setup
    data, metadata_dict = load_dataset('multi_table', 'fake_hotels', limit_dataset_size=False)
    clavaddpm_instance = ClavaDDPMSynthesizer()
    ClavaDDPMSynthesizer._MODEL_KWARGS = {
        'num_clusters': 5,
        'clustering_method': 'both',
        'd_layers': (128, 128),
        'diffusion_iterations': 300,
        'batch_size': 512,
        'sampling_batch_size': 4096,
        'num_timesteps': 100,
        'verbose': True,
    }

    # Run
    trained_synthesizer = clavaddpm_instance.get_trained_synthesizer(data, metadata_dict)
    sampled_data = clavaddpm_instance.sample_from_synthesizer(trained_synthesizer, scale=1)

    # Assert
    for table_name, table in sampled_data.items():
        assert table.shape[1] == data[table_name].shape[1], (
            f'Sampled data shape {sampled_data.shape} does '
            f'not match original data shape {data.shape}'
        )

        assert set(table.columns) == set(data[table_name].columns)
