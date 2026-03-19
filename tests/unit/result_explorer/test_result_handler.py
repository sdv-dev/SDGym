import io
import os
import pickle
import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

from sdgym.result_explorer.result_handler import (
    LocalResultsHandler,
    ResultsHandler,
    S3ResultsHandler,
)
from sdgym.run_benchmark.utils import TIMEOUT


class TestResultsHandler:
    """Unit tests for the ResultsHandler class."""

    def test__validate_folder_name(self):
        """Test the `_validate_folder_name` method."""
        # Setup
        handler = Mock()
        handler.list = Mock(return_value=['run1', 'run2'])
        expected_error = re.escape("Folder 'run3' does not exist in the results directory.")

        # Run and Assert
        ResultsHandler._validate_folder_name(handler, 'run1')
        with pytest.raises(ValueError, match=expected_error):
            ResultsHandler._validate_folder_name(handler, 'run3')

    def test__compute_pareto_frontier_dataset(self):
        """Test the `_compute_pareto_frontier_dataset` method."""
        # Setup
        dataset_results = pd.DataFrame(
            {
                'Adjusted_Quality_Score': [0.90, 0.80, 0.85, 0.80],
                'Adjusted_Total_Time': [10, 20, 15, 10],
            },
            index=[10, 11, 12, 13],
        )
        handler = LocalResultsHandler(base_path='.')

        # Run
        result = handler._compute_pareto_frontier_dataset(dataset_results)

        # Assert
        expected_result = pd.Series([True, False, False, True], index=[10, 11, 12, 13])
        pd.testing.assert_series_equal(result, expected_result)

    def test__compute_pareto_frontier(self):
        """Test the `_compute_pareto_frontier` method."""
        # Setup
        data = pd.DataFrame(
            {
                'Dataset': ['A', 'A', 'B', 'B'],
                'Synthesizer': ['Synth1', 'Synth2', 'Synth1', 'Synth2'],
                'Adjusted_Quality_Score': [0.5, 0.6, 0.7, 0.8],
                'Adjusted_Total_Time': [10, 20, 30, 40],
            },
            index=[4, 5, 6, 7],
        )
        handler = LocalResultsHandler(base_path='.')
        handler._compute_pareto_frontier_dataset = Mock(
            side_effect=[
                pd.Series([True, False], index=[4, 5]),
                pd.Series([False, True], index=[6, 7]),
            ]
        )

        # Run
        result = handler._compute_pareto_frontier(data)

        # Assert
        expected_result = pd.Series([True, False, False, True], index=[4, 5, 6, 7])
        pd.testing.assert_series_equal(result, expected_result)

        assert handler._compute_pareto_frontier_dataset.call_count == 2
        pd.testing.assert_frame_equal(
            handler._compute_pareto_frontier_dataset.call_args_list[0].args[0],
            data.iloc[[0, 1]],
        )
        pd.testing.assert_frame_equal(
            handler._compute_pareto_frontier_dataset.call_args_list[1].args[0],
            data.iloc[[2, 3]],
        )

    def test__compute_meets_baseline_quality(self):
        """Test the `_compute_meets_baseline_quality` method."""
        # Setup
        data = pd.DataFrame({
            'Dataset': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Synthesizer': [
                'GaussianCopulaSynthesizer',
                'Synth1',
                'Synth2',
                'GaussianCopulaSynthesizer',
                'Synth1',
                'Synth2',
            ],
            'Adjusted_Quality_Score': [0.90, 0.80, 0.95, 0.70, 0.70, 0.60],
            'Adjusted_Total_Time': [10, 20, 15, 100, 90, 80],
        })
        handler = LocalResultsHandler(base_path='.')

        # Run
        result = handler._compute_meets_baseline_quality(data)

        # Assert
        expected_result = pd.Series([True, False, True, True, True, False])
        pd.testing.assert_series_equal(result, expected_result)

    def test__compute_wins_mock(self):
        """Test the `_compute_wins` method with mocks."""
        # Setup
        data = pd.DataFrame({
            'Dataset': ['A', 'A', 'B', 'B', 'C'],
            'Synthesizer': ['Synth1', 'Synth2', 'Synth1', 'Synth2', 'Synth1'],
            'Adjusted_Quality_Score': [0.5, 0.6, 0.7, 0.6, 0.8],
            'Adjusted_Total_Time': [10, 20, 150, 100, 220],
        })
        handler = LocalResultsHandler(base_path='.')
        handler._compute_meets_baseline_quality = Mock(
            return_value=pd.Series([True, False, True, False, True])
        )
        handler._compute_pareto_frontier = Mock(
            return_value=pd.Series([True, True, True, False, True])
        )

        # Run
        result = handler._compute_wins(data)

        # Assert
        expected_result = pd.DataFrame({
            'Dataset': ['A', 'A', 'B', 'B', 'C'],
            'Synthesizer': ['Synth1', 'Synth2', 'Synth1', 'Synth2', 'Synth1'],
            'Adjusted_Quality_Score': [0.5, 0.6, 0.7, 0.6, 0.8],
            'Adjusted_Total_Time': [10, 20, 150, 100, 220],
            'Win': pd.Series([1, 0, 1, 0, 1], dtype=result['Win'].dtype),
        })
        pd.testing.assert_frame_equal(result, expected_result)
        handler._compute_meets_baseline_quality.assert_called_once_with(data)
        handler._compute_pareto_frontier.assert_called_once_with(data)

    def test__compute_wins(self):
        """Test the `_compute_wins` method.

        The expected result is:
        - For dataset `A`: only `GaussianCopulaSynthesizer` wins because it meets the
        baseline quality and strictly dominates the other synthesizers.
        - For dataset `B`: only `Synth1` wins because `GaussianCopulaSynthesizer` is
        dominated, and `Synth2` is on the Pareto frontier but does not meet the
        baseline quality.
        - For dataset `C`: `GaussianCopulaSynthesizer` and `Synth1` win because they
        both meet the baseline quality, and equal quality with lower time does not
        strictly dominate.
        - For dataset `D`: all synthesizers win because they all meet the baseline
        quality and none is strictly dominated.
        - For dataset `E`: `GaussianCopulaSynthesizer` and `Synth1` win because exact
        ties do not strictly dominate, while `Synth2` does not meet the baseline
        quality.
        """
        # Setup
        # fmt: off
        data = pd.DataFrame({
            'Dataset': [
                'A', 'A', 'A',
                'B', 'B', 'B',
                'C', 'C', 'C',
                'D', 'D', 'D',
                'E', 'E', 'E',
            ],
            'Synthesizer': [
                'GaussianCopulaSynthesizer', 'Synth1', 'Synth2',
                'GaussianCopulaSynthesizer', 'Synth1', 'Synth2',
                'GaussianCopulaSynthesizer', 'Synth1', 'Synth2',
                'GaussianCopulaSynthesizer', 'Synth1', 'Synth2',
                'GaussianCopulaSynthesizer', 'Synth1', 'Synth2',
            ],
            'Adjusted_Quality_Score': [
                0.90, 0.80, 0.85,
                0.70, 0.75, 0.60,
                0.80, 0.80, 0.79,
                0.60, 0.70, 0.65,
                0.82, 0.82, 0.81,
            ],
            'Adjusted_Total_Time': [
                10, 20, 15,
                150, 100, 90,
                220, 210, 200,
                300, 340, 320,
                110, 110, 105,
            ],
        })
        # fmt: on
        handler = LocalResultsHandler(base_path='.')

        # Run
        result = handler._compute_wins(data)

        # Assert
        expected_wins = {
            'A': [1, 0, 0],
            'B': [0, 1, 0],
            'C': [1, 1, 0],
            'D': [1, 1, 1],
            'E': [1, 1, 0],
        }
        expected_wins = [win for wins in expected_wins.values() for win in wins]
        assert result['Win'].tolist() == expected_wins

    def test__get_summarize_table(self):
        """Test the `_get_summarize_table` method."""
        # Setup
        folder_to_results = {
            'SDGym_results_07_15_2025': pd.DataFrame({
                'Dataset': ['A', 'B', 'C'] * 3,
                'Synthesizer': ['GaussianCopulaSynthesizer'] * 3 + ['Synth1'] * 3 + ['Synth2'] * 3,
                'Win': [0, 0, 0, 1, 1, 0, 0, 1, 0],
            })
        }
        folder_infos = {
            'SDGym_results_07_15_2025': {
                'date': '07_15_2025',
                'sdgym_version': '0.9.0',
                '# datasets': 3,
            }
        }
        handler = Mock()
        handler.baseline_synthesizer = 'GaussianCopulaSynthesizer'

        # Run
        result = ResultsHandler._get_summarize_table(handler, folder_to_results, folder_infos)

        # Assert
        expected_summary = pd.DataFrame({
            'Synthesizer': ['GaussianCopulaSynthesizer', 'Synth1', 'Synth2'],
            '07_15_2025 - # datasets: 3 - sdgym version: 0.9.0': [0, 2, 1],
        })
        pd.testing.assert_frame_equal(result, expected_summary)

    def test_get_column_name_infos(self):
        """Test the `_get_column_name_infos` method."""
        # Setup
        folder = 'SDGym_results_07_15_2025'
        yaml_content = {
            'starting_date': '07_15_2025 15:56:03',
            'sdgym_version': '0.9.0',
        }
        result = pd.DataFrame({
            'Dataset': ['A', 'B', 'C'],
            'Synthesizer': ['GaussianCopulaSynthesizer'] * 3,
        })
        folder_to_results = {folder: result}
        handler = Mock()
        handler._get_results_files = Mock(return_value=['run_config.yaml'])
        handler._load_yaml_file = Mock(return_value=yaml_content)
        handler.baseline_synthesizer = 'GaussianCopulaSynthesizer'

        # Run
        info = ResultsHandler._get_column_name_infos(handler, folder_to_results)

        # Assert
        assert info == {folder: {'date': '07_15_2025', 'sdgym_version': '0.9.0', '# datasets': 3}}

    def test__process_results(self):
        """Test the `_process_results` method."""
        # Setup
        results = [
            pd.DataFrame({
                'Dataset': ['A', 'A', 'A', 'B', 'B', 'B'],
                'Synthesizer': [
                    'Synth1',
                    'Synth2(1)',
                    'UniformSynthesizer',
                    'Synth1',
                    'Synth2(1)',
                    'UniformSynthesizer',
                ],
                'Train_Time': [10, 100, 1000, 20, 200, 2000],
                'Sample_Time': [1, 10, 100, 2, 20, 200],
                'Quality_Score': [0.1, 0.4, 0.7, 0.2, 0.5, 0.8],
                'Error': [None, 'Synthesizer Timeout', None, None, None, None],
                'Adjusted_Total_Time': [1011, 1000 + TIMEOUT + 100, 2100, 2022, 2220, 4200],
                'Adjusted_Quality_Score': [0.1, 0.7, 0.7, 0.2, 0.5, 0.8],
            }),
            pd.DataFrame({
                'Dataset': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
                'Synthesizer': [
                    'UniformSynthesizer(2)',
                    'Synth1(2)',
                    'UniformSynthesizer(3)',
                    'Synth2(2)',
                    'Synth1',
                    'Synth2',
                    'UniformSynthesizer',
                ],
                'Train_Time': [9999, 111, 9998, 222, 30, 300, 3000],
                'Sample_Time': [999, 11, 998, 22, 3, 30, 300],
                'Quality_Score': [0.99, 0.11, 0.98, 0.22, 0.3, 0.6, 0.9],
                'Error': [None, None, None, None, None, None, None],
                'Adjusted_Total_Time': [10998, 10121, 10996, 10242, 3033, 3330, 6300],
                'Adjusted_Quality_Score': [0.99, 0.11, 0.98, 0.22, 0.3, 0.6, 0.9],
            }),
        ]
        invalid_results = [
            pd.DataFrame({
                'Dataset': ['A', 'A', 'B', 'B'],
                'Synthesizer': ['Synth1', 'Synth2', 'Synth1', 'UniformSynthesizer'],
                'Train_Time': [10, 100, 20, 2000],
                'Sample_Time': [1, 10, 2, 200],
                'Quality_Score': [0.1, 0.4, 0.2, 0.8],
                'Adjusted_Total_Time': [11, 110, 22, 2200],
                'Adjusted_Quality_Score': [0.1, 0.4, 0.2, 0.8],
            }),
        ]
        handler = Mock()
        expected_error_message = re.escape(
            'There is no dataset that has been run by all synthesizers. Cannot summarize results.'
        )

        # Run
        processed_results = ResultsHandler._process_results(handler, results)
        with pytest.raises(ValueError, match=expected_error_message):
            ResultsHandler._process_results(handler, invalid_results)

        # Assert
        expected_results = pd.DataFrame({
            'Dataset': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'Synthesizer': [
                'Synth1',
                'Synth2',
                'UniformSynthesizer',
                'Synth1',
                'Synth2',
                'UniformSynthesizer',
                'Synth1',
                'Synth2',
                'UniformSynthesizer',
            ],
            'Train_Time': [10, 100, 1000, 20, 200, 2000, 30, 300, 3000],
            'Sample_Time': [1, 10, 100, 2, 20, 200, 3, 30, 300],
            'Quality_Score': [0.1, 0.4, 0.7, 0.2, 0.5, 0.8, 0.3, 0.6, 0.9],
            'Error': [None, 'Synthesizer Timeout', None, None, None, None, None, None, None],
            'Adjusted_Total_Time': [
                1011,
                1000 + TIMEOUT + 100,
                2100,
                2022,
                2220,
                4200,
                3033,
                3330,
                6300,
            ],
            'Adjusted_Quality_Score': [0.1, 0.7, 0.7, 0.2, 0.5, 0.8, 0.3, 0.6, 0.9],
        })
        pd.testing.assert_frame_equal(processed_results, expected_results, check_dtype=False)

    def test_summarize(self):
        """Test the `summarize` method."""
        # Setup
        folder_name = 'SDGym_results_07_15_2025'
        handler = Mock()
        handler.list = Mock(return_value=[folder_name])
        handler._get_results_files = Mock(return_value=['results.csv', 'results(1).csv'])
        result_1 = pd.DataFrame({
            'Dataset': ['A', 'B'],
            'Synthesizer': ['Synth1'] * 2,
            'Quality_Score': [0.5, 0.6],
        })
        result_2 = pd.DataFrame({
            'Dataset': ['A', 'B'],
            'Synthesizer': ['Synth2'] * 2,
            'Quality_Score': [0.7, 0.8],
        })
        result_list = [result_1, result_2]
        handler._get_results = Mock(return_value=result_list)
        aggregated_results = pd.concat(result_list, ignore_index=True)
        handler._compute_wins = Mock(return_value=aggregated_results)
        handler._get_column_name_infos = Mock(
            return_value={
                folder_name: {'date': '07_15_2025', 'sdgym_version': '0.9.0', '# datasets': 1}
            }
        )
        result = pd.DataFrame({
            '07_15_2025 - # datasets: 1 - sdgym version: 0.9.0': [1],
            'Synthesizer': ['Synth1'],
        }).set_index('Synthesizer')
        handler._get_summarize_table = Mock(return_value=result)
        handler._process_results = Mock(return_value=aggregated_results)

        # Run
        summary, benchmark_result = ResultsHandler.summarize(handler, folder_name)

        # Assert
        pd.testing.assert_frame_equal(summary, result)
        pd.testing.assert_frame_equal(benchmark_result, aggregated_results)
        handler.list.assert_called_once()
        handler._get_results_files.assert_called_once_with(
            folder_name, prefix='results', suffix='.csv'
        )
        handler._get_results.assert_called_once_with(folder_name, ['results.csv', 'results(1).csv'])
        handler._process_results.assert_called_once_with(result_list)
        compute_wing_args = handler._compute_wins.call_args[0][0]
        pd.testing.assert_frame_equal(compute_wing_args, aggregated_results)
        _get_column_name_infos_args = handler._get_column_name_infos.call_args[0][0]
        for folder, agg_result in _get_column_name_infos_args.items():
            assert folder == folder_name
            pd.testing.assert_frame_equal(agg_result, aggregated_results)

    @pytest.mark.parametrize(
        'dataset_names, synthesizer_names, summary, expected_error_message',
        [
            (None, None, False, None),
            (['A'], None, True, None),
            (None, ['Synth1'], False, None),
            ('A', None, False, re.escape('`dataset_names` must be a list of strings or None.')),
            (
                None,
                'Synth1',
                False,
                re.escape('`synthesizer_names` must be a list of strings or None.'),
            ),
            (['A'], ['Synth1'], 'not_a_bool', re.escape('`summary` must be a boolean.')),
        ],
    )
    def test__validate_load_results_filters(
        self, dataset_names, synthesizer_names, summary, expected_error_message
    ):
        """Test the `_validate_load_results_filters` method."""
        # Setup
        handler = Mock()

        # Run and Assert
        if expected_error_message is not None:
            with pytest.raises(ValueError, match=expected_error_message):
                ResultsHandler._validate_load_results_filters(
                    handler, dataset_names, synthesizer_names, summary
                )
        else:
            ResultsHandler._validate_load_results_filters(
                handler, dataset_names, synthesizer_names, summary
            )

    def test_load_results(self):
        """Test the `load_results` method."""
        # Setup
        folder_name = 'SDGym_results_07_15_2025'
        handler = Mock()
        handler._validate_folder_name = Mock()
        handler._validate_load_results_filters = Mock()
        handler._get_results_files = Mock(return_value=['results.csv', 'results(1).csv'])
        result_1 = pd.DataFrame({
            'Dataset': ['A', 'B'],
            'Synthesizer': ['Synth1'] * 2,
            'Quality_Score': [0.5, 0.6],
        })
        result_2 = pd.DataFrame({
            'Dataset': ['A', 'B'],
            'Synthesizer': ['Synth2'] * 2,
            'Quality_Score': [0.7, 0.8],
        })
        result_list = [result_1, result_2]
        handler._get_results = Mock(return_value=result_list)

        # Run
        results = ResultsHandler.load_results(handler, folder_name)

        # Assert
        handler._validate_folder_name.assert_called_once_with(folder_name)
        handler._validate_load_results_filters.assert_called_once_with(None, None, False)
        expected_results = pd.concat(result_list, ignore_index=True)
        pd.testing.assert_frame_equal(results, expected_results)
        handler._get_results_files.assert_called_once_with(
            folder_name, prefix='results', suffix='.csv'
        )
        handler._get_results.assert_called_once_with(folder_name, ['results.csv', 'results(1).csv'])

    @pytest.mark.parametrize(
        'dataset_names, synthesizer_names',
        [
            (['C'], ['Synth1']),
            (['A'], ['Synth3']),
            (['C'], ['Synth3']),
        ],
    )
    def test_load_results_empty(self, dataset_names, synthesizer_names):
        """Test the `load_results` method when no results are found after filtering."""
        # Setup
        folder_name = 'SDGym_results_07_15_2025'
        handler = Mock()
        handler._validate_folder_name = Mock()
        handler._validate_load_results_filters = Mock()
        handler._get_results_files = Mock(return_value=['results.csv'])
        result = pd.DataFrame({
            'Dataset': ['A', 'B'],
            'Synthesizer': ['Synth1'] * 2,
            'Quality_Score': [0.5, 0.6],
        })
        handler._get_results = Mock(return_value=[result])
        filters = []
        if dataset_names is not None:
            filters.append(f'- Datasets: {", ".join(dataset_names)}')
        if synthesizer_names is not None:
            filters.append(f'- Synthesizers: {", ".join(synthesizer_names)}')

        filters_text = '\n'.join(filters)

        expected_warning_message = re.escape(
            f'No results found in folder "{folder_name}" '
            f'matching the specified filters:\n'
            f'{filters_text}'
        )

        # Run and Assert
        with pytest.warns(UserWarning, match=expected_warning_message):
            ResultsHandler.load_results(
                handler,
                folder_name,
                dataset_names,
                synthesizer_names,
            )

    def test_load_metainfo(self):
        """Test the `load_metainfo` method."""
        # Setup
        folder_name = 'SDGym_results_07_15_2025'
        handler = Mock()
        handler._validate_folder_name = Mock()
        handler._get_results_files = Mock(return_value=['metainfo.yaml', 'metainfo(1).yaml'])
        yaml_content_1 = {'run_id': 'run_1', 'sdgym_version': '0.9.0'}
        yaml_content_2 = {'run_id': 'run_2', 'sdgym_version': '0.9.1'}
        handler._load_yaml_file = Mock(side_effect=[yaml_content_1, yaml_content_2])

        # Run
        metainfo = ResultsHandler.load_metainfo(handler, folder_name)

        # Assert
        handler._validate_folder_name.assert_called_once_with(folder_name)
        expected_metainfo = {
            'run_1': {'sdgym_version': '0.9.0'},
            'run_2': {'sdgym_version': '0.9.1'},
        }
        assert metainfo == expected_metainfo
        handler._get_results_files.assert_called_once_with(
            folder_name, prefix='metainfo', suffix='.yaml'
        )
        assert handler._load_yaml_file.call_count == 2


class TestLocalResultsHandler:
    """Unit tests for the LocalResultsHandler class."""

    def test__init__sets_base_path_and_default_baseline(self, tmp_path):
        """Test it initializes base_path and default baseline."""
        # Run
        handler = LocalResultsHandler(str(tmp_path))

        # Assert
        assert handler.base_path == str(tmp_path)
        assert handler.baseline_synthesizer == 'GaussianCopulaSynthesizer'

    def test__init__supports_baseline_override(self, tmp_path):
        """Test it allows overriding baseline synthesizer."""
        # Run
        handler = LocalResultsHandler(str(tmp_path), baseline_synthesizer='CustomBaseline')

        # Assert
        assert handler.base_path == str(tmp_path)
        assert handler.baseline_synthesizer == 'CustomBaseline'

    def test_list(self, tmp_path):
        """Test the `list` method"""
        # Setup
        path = tmp_path / 'results'
        path.mkdir()
        (path / 'run1').mkdir()
        (path / 'run2').mkdir()
        result_handler = LocalResultsHandler(str(path))

        # Run
        runs = result_handler.list()

        # Assert
        assert set(runs) == {'run1', 'run2'}

    def test_load_synthetic_data(self, tmp_path):
        """Test the `load_synthetic_data` method"""
        # Setup
        path = (
            tmp_path
            / 'results'
            / 'run1'
            / 'expedia_hotel_logs_07_07_2025'
            / 'GaussianCopulaSynthesizer'
        )
        path.mkdir(parents=True)
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        data.to_csv(path / 'synthetic_data.csv', index=False)
        file_path = str(path / 'synthetic_data.csv')
        result_handler = LocalResultsHandler(str(tmp_path))

        # Run
        synthetic_data = result_handler.load_synthetic_data(file_path)

        # Assert
        pd.testing.assert_frame_equal(synthetic_data, data)

    def test_load_synthesizer(self, tmp_path):
        """Test the `load_synthesizer` method"""
        # Setup
        synthesizer = GaussianCopulaSynthesizer(Metadata())
        folder_name = 'SDGym_results_07_07_2025'
        dataset_name = 'adult'
        synthesizer_name = 'GaussianCopulaSynthesizer'
        (tmp_path / folder_name).mkdir(parents=True, exist_ok=True)
        (tmp_path / folder_name / f'{dataset_name}_07_07_2025').mkdir(parents=True, exist_ok=True)
        (tmp_path / folder_name / f'{dataset_name}_07_07_2025' / synthesizer_name).mkdir(
            parents=True, exist_ok=True
        )
        synthesizer_path = (
            tmp_path
            / folder_name
            / f'{dataset_name}_07_07_2025'
            / synthesizer_name
            / f'{synthesizer_name}.pkl'
        )
        synthesizer.save(synthesizer_path)
        result_handler = LocalResultsHandler(str(tmp_path))

        # Run
        loaded_synthesizer = result_handler.load_synthesizer(str(synthesizer_path))

        # Assert
        assert loaded_synthesizer is not None
        assert isinstance(loaded_synthesizer, GaussianCopulaSynthesizer)

    def test_load_synthetic_data_zip(self, tmp_path):
        """Test the `load_synthetic_data` method for zipped multi-table data (local)."""
        # Setup
        base = tmp_path / 'results'
        data_dir = base / 'SDGym_results_07_07_2025' / 'dataset_07_07_2025' / 'Synth'
        data_dir.mkdir(parents=True)

        # Create a zip with two csvs
        import zipfile

        zip_path = data_dir / 'Synth_synthetic_data.zip'
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('table1.csv', 'a,b\n1,2\n')
            zf.writestr('table2.csv', 'x,y\n3,4\n')

        result_handler = LocalResultsHandler(str(base))

        # Run
        tables = result_handler.load_synthetic_data(str(zip_path))

        # Assert
        assert set(tables.keys()) == {'table1', 'table2'}
        pd.testing.assert_frame_equal(tables['table1'], pd.DataFrame({'a': [1], 'b': [2]}))
        pd.testing.assert_frame_equal(tables['table2'], pd.DataFrame({'x': [3], 'y': [4]}))

    @patch('os.path.exists')
    @patch('os.path.isfile')
    def test_get_file_path_local(self, mock_isfile, mock_exists):
        """Test `get_file_path` when files exist."""
        # Setup
        handler = Mock()
        handler.base_path = '/local/results'
        path_parts = ['results_folder_07_07_2025', 'my_dataset']
        mock_exists.return_value = True
        mock_isfile.return_value = True

        # Run
        file_path = LocalResultsHandler.get_file_path(handler, path_parts, 'synthesizer.pkl')

        # Assert
        expected_file_path = os.path.join(
            'results_folder_07_07_2025', 'my_dataset', 'synthesizer.pkl'
        )
        assert file_path == expected_file_path
        mock_exists.assert_called_once_with(
            os.path.join(
                handler.base_path,
                'results_folder_07_07_2025',
                'my_dataset',
            )
        )
        mock_isfile.assert_called_once_with(
            os.path.join(
                handler.base_path,
                'results_folder_07_07_2025',
                'my_dataset',
                'synthesizer.pkl',
            )
        )

    @patch('os.path.exists')
    @patch('os.path.isfile')
    def test_get_file_path_local_error(self, mock_isfile, mock_exists):
        """Test `get_file_path` for local path when files do not exist."""
        # Setup
        handler = Mock()
        handler.base_path = '/local/results'
        results_folder_name = 'SDGym_results_07_07_2025'
        dataset_name = 'adult'
        mock_exists.return_value = False
        synthesizer_name = 'GaussianCopulaSynthesizer'
        synthesizer_path = os.path.join(
            handler.base_path, results_folder_name, f'{dataset_name}_07_07_2025', synthesizer_name
        )
        path_parts = [results_folder_name, f'{dataset_name}_07_07_2025', synthesizer_name]
        error_message_1 = re.escape(f'Path does not exist: {synthesizer_path}')
        error_message_2 = re.escape('File does not exist: synthesizer.pkl')

        # Run and Assert
        with pytest.raises(ValueError, match=error_message_1):
            LocalResultsHandler.get_file_path(handler, path_parts, 'synthesizer.pkl')

        mock_exists.return_value = True
        mock_isfile.return_value = False
        with pytest.raises(ValueError, match=error_message_2):
            LocalResultsHandler.get_file_path(handler, path_parts, 'synthesizer.pkl')


class TestS3ResultsHandler:
    """Unit tests for the S3ResultsHandler class."""

    def test__init__(self):
        """Test the `__init__` method."""
        # Setup
        path = 's3://my-bucket/prefix'

        # Run
        result_handler = S3ResultsHandler(path, 's3_client')

        # Assert
        assert result_handler.s3_client == 's3_client'
        assert result_handler.bucket_name == 'my-bucket'
        assert result_handler.prefix == 'prefix/'
        assert result_handler.baseline_synthesizer == 'GaussianCopulaSynthesizer'

    def test__init__supports_baseline_override(self):
        """Test it allows overriding baseline synthesizer."""
        # Run
        s3_client = Mock()
        handler = S3ResultsHandler(
            's3://bkt/prefix', s3_client, baseline_synthesizer='CustomBaseline'
        )

        # Assert
        assert handler.baseline_synthesizer == 'CustomBaseline'
        assert handler.s3_client == s3_client
        assert handler.bucket_name == 'bkt'
        assert handler.prefix == 'prefix/'

    def test_list(self):
        """Test the `list` method."""
        # Setup
        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {
            'CommonPrefixes': [{'Prefix': 'run1/'}, {'Prefix': 'run2/'}]
        }
        result_handler = Mock()
        result_handler.s3_client = mock_s3_client
        result_handler.bucket_name = 'my-bucket'
        result_handler.prefix = 'results/'

        # Run
        runs = S3ResultsHandler.list(result_handler)

        # Assert
        assert set(runs) == {'run1', 'run2'}
        mock_s3_client.list_objects_v2.assert_called_once_with(
            Bucket='my-bucket', Prefix='results/', Delimiter='/'
        )

    def test_load_synthetic_data(self):
        """Test the `load_synthetic_data` method."""
        # Setup
        mock_s3_client = Mock()
        mock_s3_client.get_object.return_value = {'Body': Mock(read=lambda: b'col1,col2\n1,3\n2,4')}
        result_handler = S3ResultsHandler('s3://my-bucket/prefix', mock_s3_client)
        result_handler.s3_client = mock_s3_client

        # Run
        synthetic_data = result_handler.load_synthetic_data('synthetic_data.csv')

        # Assert
        expected_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        pd.testing.assert_frame_equal(synthetic_data, expected_data)
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='my-bucket', Key='prefix/synthetic_data.csv'
        )

    def test_load_synthesizer(self):
        """Test the `load_synthesizer` method."""
        # Setup
        mock_s3_client = Mock()
        synthesizer = GaussianCopulaSynthesizer(Metadata())
        mock_s3_client.get_object.return_value = {
            'Body': Mock(read=lambda: pickle.dumps(synthesizer))
        }
        result_handler = S3ResultsHandler('s3://my-bucket/prefix', mock_s3_client)
        result_handler.s3_client = mock_s3_client

        # Run
        loaded_synthesizer = result_handler.load_synthesizer('synthesizer.pkl')

        # Assert
        assert isinstance(loaded_synthesizer, GaussianCopulaSynthesizer)
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='my-bucket', Key='prefix/synthesizer.pkl'
        )

    def test_load_synthetic_data_zip(self):
        """Test the `load_synthetic_data` method for zipped multi-table data (S3)."""
        # Setup
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('customers.csv', 'id,age\n1,30\n')
            zf.writestr('transactions.csv', 'id,amount\n1,100\n')
        buffer.seek(0)

        mock_s3_client = Mock()
        mock_s3_client.get_object.return_value = {'Body': Mock(read=lambda: buffer.getvalue())}
        result_handler = S3ResultsHandler('s3://my-bucket/prefix', mock_s3_client)

        # Run
        tables = result_handler.load_synthetic_data('some/path.zip')

        # Assert
        assert set(tables.keys()) == {'customers', 'transactions'}
        pd.testing.assert_frame_equal(tables['customers'], pd.DataFrame({'id': [1], 'age': [30]}))
        pd.testing.assert_frame_equal(
            tables['transactions'], pd.DataFrame({'id': [1], 'amount': [100]})
        )
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='my-bucket', Key='prefix/some/path.zip'
        )

    def test_get_file_path_s3(self):
        """Test `get_file_path` for S3 path when folders and file exist."""
        # Setup
        mock_s3_client = Mock()
        handler = S3ResultsHandler('s3://my-bucket/prefix', mock_s3_client)
        path_parts = ['results_folder_07_07_2025', 'my_dataset']
        end_filename = 'synthesizer.pkl'
        file_path = 'results_folder_07_07_2025/my_dataset/synthesizer.pkl'
        mock_s3_client.list_objects_v2.return_value = {'Contents': [{}]}

        # Run
        result = handler.get_file_path(path_parts, end_filename)

        # Assert
        assert result == file_path
        mock_s3_client.list_objects_v2.assert_any_call(
            Bucket='my-bucket', Prefix='prefix/results_folder_07_07_2025/', MaxKeys=1
        )
        mock_s3_client.list_objects_v2.assert_any_call(
            Bucket='my-bucket', Prefix='prefix/results_folder_07_07_2025/my_dataset/', MaxKeys=1
        )
        mock_s3_client.head_object.assert_called_once_with(
            Bucket='my-bucket', Key='prefix/results_folder_07_07_2025/my_dataset/synthesizer.pkl'
        )
