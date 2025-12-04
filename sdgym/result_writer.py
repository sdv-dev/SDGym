"""Results writer for SDGym benchmark."""

import io
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import cloudpickle
import pandas as pd
import plotly.graph_objects as go
import yaml
from openpyxl import load_workbook

from sdgym.s3 import parse_s3_path


class ResultsWriter(ABC):
    """Abstract base class for writing results to files."""

    @abstractmethod
    def write_dataframe(self, data, file_path, append=False):
        """Write a DataFrame to a file."""
        pass

    @abstractmethod
    def write_pickle(self, obj, file_path):
        """Write a Python object to a pickle file."""
        pass

    @abstractmethod
    def write_yaml(self, data, file_path, append=False):
        """Write data to a YAML file."""
        pass


class LocalResultsWriter:
    """Local results writer for saving results to the local filesystem."""

    def write_zipped_dataframes(self, data, file_path, index=False):
        """Write a dictoinary of dataframes to a ZIP file."""
        with zipfile.ZipFile(file_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            for table_name, table in data.items():
                buf = io.StringIO()
                table.to_csv(buf, index=index)
                zf.writestr(f'{table_name}.csv', buf.getvalue())

    def write_dataframe(self, data, file_path, append=False, index=False):
        """Write a DataFrame to a CSV file."""
        file_path = Path(file_path)
        if file_path.exists() and append:
            data.to_csv(file_path, mode='a', index=index, header=False)
        else:
            data.to_csv(file_path, mode='w', index=index, header=True)

    def process_data(self, writer, file_path, temp_images, sheet_name, obj, index=False):
        """Process a data item (DataFrame or Figure) and write it to the Excel writer."""
        if isinstance(obj, pd.DataFrame):
            obj.to_excel(writer, sheet_name=sheet_name, index=index)
        elif isinstance(obj, go.Figure):
            img_path = file_path.parent / f'{sheet_name}.png'
            obj.write_image(img_path)
            temp_images[sheet_name] = img_path

    def write_xlsx(self, data, file_path, index=False):
        """Write DataFrames to an Excel file.

        - Each DataFrame is saved as a table in its own sheet.
        - Newly written sheets are moved to the front.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            writer = pd.ExcelWriter(
                file_path, mode='a', engine='openpyxl', if_sheet_exists='replace'
            )
        else:
            writer = pd.ExcelWriter(file_path, mode='w', engine='openpyxl')

        with writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=index)

        wb = load_workbook(file_path)
        for sheet_name in reversed(data.keys()):
            ws = wb[sheet_name]
            wb._sheets.remove(ws)
            wb._sheets.insert(0, ws)

        wb.save(file_path)

    def write_pickle(self, obj, file_path):
        """Write a Python object to a pickle file."""
        with open(file_path, 'wb') as f:
            cloudpickle.dump(obj, f)

    def write_yaml(self, data, file_path, append=False):
        """Write data to a YAML file."""
        file_path = Path(file_path)
        if append:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    run_data = yaml.safe_load(f) or {}
                for key, value in data.items():
                    run_data[key] = value

                data = run_data

        with open(file_path, 'w') as f:
            yaml.dump(data, f)


class S3ResultsWriter(ResultsWriter):
    """Results writer for S3."""

    def __init__(self, s3_client):
        self.s3_client = s3_client

    def write_dataframe(self, data, file_path, append=False, index=False):
        """Write a DataFrame to S3 as a CSV file."""
        bucket, key = parse_s3_path(file_path)
        if append:
            try:
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                existing_data = pd.read_csv(io.BytesIO(response['Body'].read()))
                if not existing_data.empty:
                    data = pd.concat([existing_data, data], ignore_index=True)

            except Exception:
                pass  # If the file does not exist, we will create it

        csv_buffer = data.to_csv(index=index).encode()
        self.s3_client.put_object(Body=csv_buffer, Bucket=bucket, Key=key)

    def write_pickle(self, obj, file_path):
        """Write a Python object to S3 as a pickle file."""
        bucket, key = parse_s3_path(file_path)
        buffer = io.BytesIO()
        cloudpickle.dump(obj, buffer)
        buffer.seek(0)
        self.s3_client.put_object(Body=buffer.read(), Bucket=bucket, Key=key)

    def write_yaml(self, data, file_path, append=False):
        """Write data to a YAML file in S3."""
        bucket, key = parse_s3_path(file_path)
        run_data = {}
        if append:
            try:
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read().decode()
                run_data = yaml.safe_load(content) or {}
            except self.s3_client.exceptions.NoSuchKey:
                pass

        run_data.update(data)
        new_content = yaml.dump(run_data)
        self.s3_client.put_object(Body=new_content.encode(), Bucket=bucket, Key=key)

    def write_zipped_dataframes(self, data, file_path, index=False):
        """Write a dictionary of DataFrames to a ZIP file in S3."""
        bucket, key = parse_s3_path(file_path)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            for table_name, table in data.items():
                csv_buf = io.StringIO()
                table.to_csv(csv_buf, index=index)
                zf.writestr(f'{table_name}.csv', csv_buf.getvalue())

        zip_buffer.seek(0)
        self.s3_client.upload_fileobj(zip_buffer, bucket, key)
