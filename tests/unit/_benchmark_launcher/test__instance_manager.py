"""Unit tests for the instance manager classes."""

from unittest.mock import Mock, call, patch

import pytest

from sdgym._benchmark_launcher._instance_manager import (
    BaseInstanceManager,
    GCPInstanceManager,
)


class TestBaseInstanceManager:
    def test_list_instances(self):
        """Test the `list_instances` method."""
        # Setup
        instance_manager = BaseInstanceManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            instance_manager.list_instances()

    def test_update_instance_statuses(self):
        """Test the `update_instance_statuses` method."""
        # Setup
        instance_manager = BaseInstanceManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            instance_manager.update_instance_statuses(['instance-1'], {'instance-1': 'running'})

    def test_terminate_instances(self):
        """Test the `terminate_instances` method."""
        # Setup
        instance_manager = BaseInstanceManager()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            instance_manager.terminate_instances(['instance-1'], verbose=True)


class TestGCPInstanceManager:
    def test__init__(self):
        """Test the `__init__` method."""
        # Setup
        credentials_filepath = 'creds.json'

        # Run
        instance_manager = GCPInstanceManager(credentials_filepath)

        # Assert
        assert instance_manager.credentials_filepath == credentials_filepath

    @patch('sdgym._benchmark_launcher._instance_manager.resolve_credentials')
    @patch(
        'sdgym._benchmark_launcher._instance_manager.service_account.Credentials.'
        'from_service_account_info'
    )
    @patch('sdgym._benchmark_launcher._instance_manager.compute_v1.InstancesClient')
    @patch('sdgym._benchmark_launcher._instance_manager._validate_gcp_credentials')
    def test_get_client(
        self,
        mock_validate_gcp_credentials,
        mock_instances_client,
        mock_from_service_account_info,
        mock_resolve_credentials,
    ):
        """Test the `_get_client` method."""
        # Setup
        instance_manager = GCPInstanceManager('creds.json')
        mock_resolve_credentials.return_value = {
            'gcp': {
                'project_id': 'test-project',
                'client_email': 'test@test.com',
                'token_uri': 'https://oauth2.googleapis.com/token',
                'private_key_id': 'key-id',
                'private_key': '-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n',
            }
        }
        mock_validate_gcp_credentials.return_value = []
        mock_credentials = Mock()
        mock_from_service_account_info.return_value = mock_credentials
        mock_client = Mock()
        mock_instances_client.return_value = mock_client

        # Run
        result_client, result_project_id = instance_manager._get_client()

        # Assert
        mock_resolve_credentials.assert_called_once_with('creds.json')
        mock_validate_gcp_credentials.assert_called_once_with(mock_resolve_credentials.return_value)
        mock_from_service_account_info.assert_called_once_with(
            mock_resolve_credentials.return_value['gcp']
        )
        mock_instances_client.assert_called_once_with(credentials=mock_credentials)
        assert result_client is mock_client
        assert result_project_id == 'test-project'

    @patch('sdgym._benchmark_launcher._instance_manager.resolve_credentials')
    @patch('sdgym._benchmark_launcher._instance_manager._validate_gcp_credentials')
    def test_get_client_invalid_credentials(
        self, mock_validate_gcp_credentials, mock_resolve_credentials
    ):
        """Test `_get_client` raises an error for invalid GCP credentials."""
        # Setup
        instance_manager = GCPInstanceManager('creds.json')
        mock_resolve_credentials.return_value = {'gcp': {}}
        mock_validate_gcp_credentials.return_value = [
            "credentials['gcp']['project_id'] is missing or empty.",
            "credentials['gcp']['private_key'] is missing or empty.",
        ]
        expected_error = (
            'Invalid GCP credentials:\n'
            "credentials['gcp']['project_id'] is missing or empty.\n"
            "credentials['gcp']['private_key'] is missing or empty."
        )

        # Run and Assert
        with pytest.raises(
            ValueError, match=expected_error.replace('[', r'\[').replace(']', r'\]')
        ):
            instance_manager._get_client()

    def test_list_instances(self):
        """Test the `list_instances` method."""
        # Setup
        instance_manager = GCPInstanceManager('creds.json')
        instance_manager._get_client = Mock(return_value=(Mock(), 'test-project'))

        running_instance = Mock()
        running_instance.id = 123
        running_instance.name = 'instance-1'
        running_instance.zone = (
            'https://www.googleapis.com/compute/v1/projects/test-project/zones/us-central1-a'
        )
        running_instance.status = 'RUNNING'

        terminated_instance = Mock()
        terminated_instance.id = 456
        terminated_instance.name = 'instance-2'
        terminated_instance.zone = (
            'https://www.googleapis.com/compute/v1/projects/test-project/zones/us-central1-b'
        )
        terminated_instance.status = 'TERMINATED'

        empty_scoped_list = Mock()
        empty_scoped_list.instances = None

        scoped_list = Mock()
        scoped_list.instances = [running_instance, terminated_instance]

        client = instance_manager._get_client.return_value[0]
        client.aggregated_list.return_value = [
            ('zones/us-central1-a', scoped_list),
            ('zones/us-central1-b', empty_scoped_list),
        ]

        # Run
        result = instance_manager.list_instances()

        # Assert
        instance_manager._get_client.assert_called_once_with()
        client.aggregated_list.assert_called_once_with(project='test-project')
        assert result == [
            {
                'id': '123',
                'name': 'instance-1',
                'zone': 'us-central1-a',
                'status': 'RUNNING',
            }
        ]

    def test_update_instance_statuses(self):
        """Test the `update_instance_statuses` method."""
        # Setup
        instance_manager = GCPInstanceManager('creds.json')
        instance_manager.list_instances = Mock(
            return_value=[
                {
                    'id': '123',
                    'name': 'instance-1',
                    'zone': 'us-central1-a',
                    'status': 'RUNNING',
                }
            ]
        )
        instance_name_to_status = {
            'instance-1': 'running',
            'instance-2': 'running',
            'instance-3': 'stopped',
        }

        # Run
        instance_manager.update_instance_statuses(
            ['instance-1', 'instance-2', 'instance-3'],
            instance_name_to_status,
        )

        # Assert
        instance_manager.list_instances.assert_called_once_with()
        assert instance_name_to_status == {
            'instance-1': 'running',
            'instance-2': 'completed',
            'instance-3': 'stopped',
        }

    @patch('builtins.print')
    def test_terminate_instances(self, mock_print):
        """Test the `terminate_instances` method."""
        # Setup
        instance_manager = GCPInstanceManager('creds.json')
        mock_client = Mock()
        instance_manager._get_client = Mock(return_value=(mock_client, 'test-project'))
        instance_manager.list_instances = Mock(
            return_value=[
                {
                    'id': '123',
                    'name': 'instance-1',
                    'zone': 'us-central1-a',
                    'status': 'RUNNING',
                },
                {
                    'id': '456',
                    'name': 'instance-2',
                    'zone': 'us-central1-b',
                    'status': 'RUNNING',
                },
            ]
        )
        mock_operation_1 = Mock()
        mock_operation_2 = Mock()
        mock_client.delete.side_effect = [mock_operation_1, mock_operation_2]

        # Run
        result = instance_manager.terminate_instances(
            ['instance-1', 'instance-2'],
            verbose=True,
        )

        # Assert
        instance_manager._get_client.assert_called_once_with()
        instance_manager.list_instances.assert_called_once_with()
        assert mock_client.delete.call_args_list == [
            call(project='test-project', zone='us-central1-a', instance='instance-1'),
            call(project='test-project', zone='us-central1-b', instance='instance-2'),
        ]
        mock_operation_1.result.assert_called_once_with()
        mock_operation_2.result.assert_called_once_with()
        assert result == ['instance-1', 'instance-2']
        mock_print.assert_has_calls([
            call("Terminating GCP instance 'instance-1' (id=123, zone=us-central1-a)..."),
            call("Terminating GCP instance 'instance-2' (id=456, zone=us-central1-b)..."),
        ])

    @patch('sdgym._benchmark_launcher._instance_manager.LOGGER')
    def test_terminate_instances_logs_not_running_instances(self, mock_logger):
        """Test `terminate_instances` logs a message for not running instances."""
        # Setup
        instance_manager = GCPInstanceManager('creds.json')
        mock_client = Mock()
        instance_manager._get_client = Mock(return_value=(mock_client, 'test-project'))
        instance_manager.list_instances = Mock(
            return_value=[
                {
                    'id': '123',
                    'name': 'instance-1',
                    'zone': 'us-central1-a',
                    'status': 'RUNNING',
                },
            ]
        )
        mock_operation = Mock()
        mock_client.delete.return_value = mock_operation

        # Run
        result = instance_manager.terminate_instances(
            ['instance-1', 'instance-2'],
            verbose=False,
        )

        # Assert
        assert result == ['instance-1']
        mock_logger.info.assert_called_once_with(
            "Some provided instance names are not currently running: 'instance-2'."
        )
        mock_client.delete.assert_called_once_with(
            project='test-project',
            zone='us-central1-a',
            instance='instance-1',
        )
        mock_operation.result.assert_called_once_with()
