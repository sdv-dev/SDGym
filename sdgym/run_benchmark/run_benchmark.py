"""Script to run a benchmark and upload results to S3."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import ClientError

from sdgym._benchmark.benchmark import (
    _benchmark_multi_table_compute_gcp,
    _benchmark_single_table_compute_gcp,
)
from sdgym._benchmark_launcher.benchmark_config import BenchmarkConfig
from sdgym.run_benchmark.utils import (
    KEY_DATE_FILE,
    _parse_args,
    get_result_folder_name,
    post_benchmark_launch_message,
)
from sdgym.s3 import get_s3_client, parse_s3_path


_METHODS = {
    ("single_table", "gcp"): _benchmark_single_table_compute_gcp,
    ("multi_table", "gcp"): _benchmark_multi_table_compute_gcp,
}


def append_benchmark_run(
    output_destination: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    date_str: str,
    modality: str,
):
    """Append a new benchmark run to the benchmark dates file in S3."""
    s3_client = get_s3_client(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    bucket, prefix = parse_s3_path(output_destination)
    key = f"{prefix}{modality}/{KEY_DATE_FILE}"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        data = json.loads(body)
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            data = {"runs": []}
        else:
            raise RuntimeError(f"Failed to read {KEY_DATE_FILE} from S3: {e}")

    data["runs"].append({"date": date_str, "folder_name": get_result_folder_name(date_str)})
    data["runs"] = sorted(data["runs"], key=lambda x: x["date"])

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data).encode("utf-8"),
    )

def _resolve_modality_config(modality): 
    """Method that resolves the config for a modality and save it into a tmp yaml file"""
    return f'benchmark_{modality}.yaml'

def main():
    args = _parse_args()
    modality = args.modality
    yaml_config = _resolve_modality_config(modality)
    config = BenchmarkConfig.load_from_yaml(yaml_config)
    config.validate()
    config.run()
    '''
    compute_service = config.compute["service"]
    method = _METHODS.get((modality, compute_service))
    credential_filepath = config.credentials['credential_filepath']
    method_kwargs = config.method_params
    method_kwargs['credential_filepath'] = credential_filepath
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    for job in config.instance_jobs:
        kwargs = {
            **method_kwargs,
            'synthesizers': job['synthesizers'],
            'sdv_datasets': job['datasets'],
        }
        method(**kwargs)

    append_benchmark_run(
        output_destination=method_kwargs['output_destination'],
        aws_access_key_id=config["credentials"]["aws_access_key_id"],
        aws_secret_access_key=config["credentials"]["aws_secret_access_key"],
        date_str=date_str,
        modality=modality,
    )
    post_benchmark_launch_message(date_str, compute_service=compute_service.upper(), modality=modality)
    '''


if __name__ == "__main__":
    main()
