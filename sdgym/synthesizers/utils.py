"""Utility functions for synthesizers in SDGym."""

from sdv.metadata import Metadata

from sdgym.synthesizers.base import BaselineSynthesizer


def get_available_single_table_synthesizers():
    """List all available single-table synthesizers in SDGym.

    Returns:
        list:
            A sorted list of available single-table synthesizer names.
    """
    return sorted(BaselineSynthesizer._get_supported_synthesizers('single_table'))


def get_available_multi_table_synthesizers():
    """List all available multi-table synthesizers in SDGym.

    Returns:
        list:
            A sorted list of available multi-table synthesizer names.
    """
    return sorted(BaselineSynthesizer._get_supported_synthesizers('multi_table'))


def _get_supported_synthesizers():
    """Get SDGym supported synthesizers.

    Returns:
        list:
            A list of available SDGym supported synthesizer names.
    """
    synthesizers = []
    for modality in ['single_table', 'multi_table']:
        synthesizers.extend(BaselineSynthesizer._get_supported_synthesizers(modality))

    return sorted(synthesizers)


def _get_single_table_name(data, metadata):
    """Get the table name from metadata or data.

    Args:
        data (dict[str, pd.DataFrame]):
            Dictionary mapping the name to the dataframe. Expected to
            be a dictionary of one dataframe.
        metadata (sdv.Metadata):
            The metadata.

    Returns:
        str:
            The name of the table.
    """
    if isinstance(metadata, Metadata):
        table_name = metadata._get_single_table_name()
    else:
        if len(data) != 1:
            raise ValueError(
                "The data contains more than one table. Could not determine 'table name'."
            )

        table_name = next(iter(data))

    return table_name
