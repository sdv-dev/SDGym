from unittest.mock import call, patch

from sdgym._run_benchmark import OUTPUT_DESTINATION_AWS
from sdgym._run_benchmark.run_benchmark import main


@patch('sdgym._run_benchmark.run_benchmark.benchmark_single_table_aws')
@patch('sdgym._run_benchmark.run_benchmark.os.getenv')
def test_main(mock_getenv, mock_benchmark_single_table_aws):
    """Test the `main` method."""
    # Setup
    mock_getenv.side_effect = ['my_access_key', 'my_secret_key']

    # Run
    main()

    # Assert
    mock_getenv.assert_any_call('AWS_ACCESS_KEY_ID')
    mock_getenv.assert_any_call('AWS_SECRET_ACCESS_KEY')
    expected_calls = []
    for synthesizer in ['GaussianCopulaSynthesizer', 'TVAESynthesizer']:
        expected_calls.append(
            call(
                output_destination=OUTPUT_DESTINATION_AWS,
                aws_access_key_id='my_access_key',
                aws_secret_access_key='my_secret_key',
                synthesizers=[synthesizer],
                sdv_datasets=['expedia_hotel_logs', 'fake_companies'],
                compute_privacy_score=False,
            )
        )

    mock_benchmark_single_table_aws.assert_has_calls(expected_calls)
