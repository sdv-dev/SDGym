from unittest.mock import patch

import pytest

from sdgym.benchmark import benchmark_single_table


@patch('sdgym.benchmark.os.path')
def test_output_file_exists(path_mock):
    """Test the benchmark function when the output path already exists."""
    # Setup
    path_mock.exists.return_value = True
    output_filepath = 'test_output.csv'

    # Run and assert
    with pytest.raises(
        ValueError,
        match='test_output.csv already exists. Please provide a file that does not already exist.',
    ):
        benchmark_single_table(
            synthesizers=['DataIdentity', 'IndependentSynthesizer', 'UniformSynthesizer'],
            sdv_datasets=['student_placements'],
            output_filepath=output_filepath
        )
