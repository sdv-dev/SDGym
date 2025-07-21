import os
from sdgym.benchmark import benchmark_single_table_aws, SDV_SINGLE_TABLE_SYNTHESIZERS
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
OUTPUT_DESTINATION_AWS = 's3://sdgym-benchmark/Debug/Issue_425'
synthesizer = SDV_SINGLE_TABLE_SYNTHESIZERS
datasets = ['expedia_hotel_logs', 'child']
    

if __name__ == '__main__':
    for synthesizer in ['GaussianCopulaSynthesizer']:
        benchmark_single_table_aws(
            output_destination=OUTPUT_DESTINATION_AWS,
            aws_access_key_id=aws_key, aws_secret_access_key=aws_secret,
            synthesizers=[synthesizer],
            sdv_datasets=datasets,
        )
