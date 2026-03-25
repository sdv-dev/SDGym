import logging

from google.cloud import compute_v1
from google.oauth2 import service_account

from sdgym._benchmark_launcher._validation import _validate_gcp_credentials
from sdgym._benchmark_launcher.utils import resolve_credentials

LOGGER = logging.getLogger(__name__)


class BaseInstanceManager:
    """Base class for compute-service-specific instance managers."""

    def list_instances(self):
        """Return non-terminated instances."""
        raise NotImplementedError

    def update_instance_statuses(self, instance_names, instance_name_to_status):
        """Update launcher-tracked instance statuses in place."""
        raise NotImplementedError

    def terminate_instances(self, instance_names, verbose):
        """Terminate instances and return deleted instance names."""
        raise NotImplementedError


class GCPInstanceManager(BaseInstanceManager):
    """Manage GCP benchmark instances."""

    def __init__(self, credentials_filepath):
        self.credentials_filepath = credentials_filepath

    def _get_client(self):
        """Build and return the GCP client and project id."""
        credentials = resolve_credentials(self.credentials_filepath)
        errors = _validate_gcp_credentials(credentials)
        if errors:
            error_message = '\n'.join(errors)
            raise ValueError(f'Invalid GCP credentials:\n{error_message}')

        project_id = credentials['gcp']['project_id']
        gcp_credentials = service_account.Credentials.from_service_account_info(credentials['gcp'])
        client = compute_v1.InstancesClient(credentials=gcp_credentials)

        return client, project_id

    def list_instances(self):
        """List all non-terminated GCP instances."""
        client, project_id = self._get_client()
        instances = []
        response = client.aggregated_list(project=project_id)
        for _, scoped_list in response:
            scoped_instances = getattr(scoped_list, 'instances', None)
            if not scoped_instances:
                continue

            for instance in scoped_instances:
                if instance.status == 'TERMINATED':
                    continue

                instances.append({
                    'id': str(instance.id),
                    'name': instance.name,
                    'zone': instance.zone.split('/')[-1],
                    'status': instance.status,
                })

        return instances

    def update_instance_statuses(self, instance_names, instance_name_to_status):
        """Update launcher-tracked instance statuses in place."""
        running_instances = self.list_instances()
        running_instance_names = {instance['name'] for instance in running_instances}
        for instance_name in instance_names:
            if instance_name in running_instance_names:
                instance_name_to_status[instance_name] = 'running'
            elif instance_name_to_status.get(instance_name) == 'running':
                instance_name_to_status[instance_name] = 'completed'

    def terminate_instances(self, instance_names, verbose):
        """Terminate GCP instances by name."""
        client, project_id = self._get_client()
        running_instances = self.list_instances()
        running_instances_by_name = {instance['name']: instance for instance in running_instances}
        instances_to_delete = [
            running_instances_by_name[name]
            for name in instance_names
            if name in running_instances_by_name
        ]

        not_running = sorted(set(instance_names) - set(running_instances_by_name))
        if not_running:
            not_running_str = "', '".join(not_running)
            LOGGER.info(
                f"Some provided instance names are not currently running: '{not_running_str}'."
            )

        deleted_instances = []
        for instance in instances_to_delete:
            if verbose:
                print(  # noqa: T201
                    f"Terminating GCP instance '{instance['name']}' "
                    f'(id={instance["id"]}, zone={instance["zone"]})...'
                )

            operation = client.delete(
                project=project_id,
                zone=instance['zone'],
                instance=instance['name'],
            )
            operation.result()
            deleted_instances.append(instance['name'])

        return deleted_instances
