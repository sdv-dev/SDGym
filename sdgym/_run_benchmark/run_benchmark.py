import os

import sdgym._run_benchmark as run_benchmark
from sdgym.benchmark import benchmark_single_table_aws

datasets = ['expedia_hotel_logs', 'fake_companies']  # DEFAULT_DATASETS


def main():
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    for synthesizer in ['GaussianCopulaSynthesizer', 'TVAESynthesizer']:
        benchmark_single_table_aws(
            output_destination=run_benchmark.OUTPUT_DESTINATION_AWS,
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            synthesizers=[synthesizer],
            sdv_datasets=datasets,
            compute_privacy_score=False,
        )


if __name__ == '__main__':
    main()
