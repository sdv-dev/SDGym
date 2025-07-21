import os

from sdgym._run_benchmark import OUTPUT_DESTINATION_AWS, RESULTS_UPLOADED
from sdgym.benchmark import SDV_SINGLE_TABLE_SYNTHESIZERS, benchmark_single_table_aws

aws_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
synthesizer = SDV_SINGLE_TABLE_SYNTHESIZERS
datasets = ['expedia_hotel_logs', 'fake_companies']


if __name__ == '__main__':
    RESULTS_UPLOADED = False
    for synthesizer in ['GaussianCopulaSynthesizer', 'TVAESynthesizer']:
        benchmark_single_table_aws(
            output_destination=OUTPUT_DESTINATION_AWS,
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            synthesizers=[synthesizer],
            sdv_datasets=datasets,
        )
