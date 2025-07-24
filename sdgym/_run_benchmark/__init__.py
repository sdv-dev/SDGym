"""Folder for the SDGym benchmark module."""

from sdgym.benchmark import SDV_SINGLE_TABLE_SYNTHESIZERS

OUTPUT_DESTINATION_AWS = 's3://sdgym-benchmark/Debug/Issue_425/'
UPLOAD_DESTINATION_AWS = 's3://sdgym-benchmark/Debug/Issue_425/'
DEBUG_SLACK_CHANNEL = 'sdv-alerts-debug'
SLACK_CHANNEL = 'sdv-alerts'
SYNTHESIZERS = SDV_SINGLE_TABLE_SYNTHESIZERS
