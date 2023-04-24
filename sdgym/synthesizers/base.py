import abc
import logging

LOGGER = logging.getLogger(__name__)


class BaselineSynthesizer(abc.ABC):
    """Base class for all the ``SDGym`` baselines."""

    @classmethod
    def get_subclasses(cls, include_parents=False):
        """Recursively find subclasses of this Baseline.

        Args:
            include_parents (bool):
                Whether to include subclasses which are parents to
                other classes. Defaults to ``False``.
        """
        subclasses = dict()
        for child in cls.__subclasses__():
            grandchildren = child.get_subclasses(include_parents)
            subclasses.update(grandchildren)
            if include_parents or not grandchildren:
                subclasses[child.__name__] = child

        return subclasses

    @classmethod
    def get_baselines(cls):
        subclasses = cls.get_subclasses(include_parents=True)
        synthesizers = []
        for _, subclass in subclasses.items():
            if abc.ABC not in subclass.__bases__:
                synthesizers.append(subclass)

        return synthesizers

    def get_trained_synthesizer(self, data, metadata):
        """Get a synthesizer that has been trained on the provided data and metadata.

        Args:
            data (pandas.DataFrame or dict):
                The data to train on.
            metadata (SingleTableMetadata or MultiTableMetadata):
                The metadata.

        Returns:
            obj:
                The synthesizer object
        """
        raise NotImplementedError()

    def sample_from_synthesizer(self, synthesizer, n_samples):
        """Sample data from the provided synthesizer.

        Args:
            synthesizer (obj):
                The synthesizer object to sample data from.
            n_samples (int):
                The number of samples to create.

        Returns:
            pandas.DataFrame or dict:
                The sampled data. If single-table, should be a DataFrame. If multi-table,
                should be a dict mapping table name to DataFrame.
        """
        return NotImplementedError()


class SingleTableBaselineSynthesizer(BaselineSynthesizer, abc.ABC):
    """Base class for all the SingleTable Baselines.

    Subclasses can choose to implement ``_fit_sample``, which will
    always be called with DataFrames and Table metadata dicts, or
    to overwrite the ``fit_sample`` method, which may be called with
    either DataFrames and Table dicts, or with dicts of tables and
    dataset metadata dicts.
    """

    def get_trained_synthesizer(self, data, metadata):
        """Get a synthesizer that has been trained on the provided data and metadata.

        Args:
            data (pandas.DataFrame):
                The data to train on.
            metadata (sdv.metadata.single_table.SingleTableMetadata):
                The metadata.

        Returns:
            obj:
                The synthesizer object
        """
        return self._get_trained_synthesizer(data, metadata)

    def sample_from_synthesizer(self, synthesizer, n_samples):
        """Sample data from the provided synthesizer.

        Args:
            synthesizer (obj):
                The synthesizer object to sample data from.
            n_samples (int):
                The number of samples to create.

        Returns:
            pandas.DataFrame:
                The sampled data.
        """
        return self._sample_from_synthesizer(synthesizer, n_samples)


class MultiSingleTableBaselineSynthesizer(BaselineSynthesizer, abc.ABC):
    """Base class for SingleTableBaselines that are used on multi table scenarios.

    These classes model and sample each table independently and then just
    randomly choose ids from the parent tables to form the relationships.

    NOTE: doesn't currently work.
    """

    def get_trained_synthesizer(self, data, metadata):
        """Get the trained synthesizer.

        Args:
            data (dict):
                A dict mapping table name to table data.
            metadata (sdv.metadata.multi_table.MultiTableMetadata):
                The multi-table metadata.

        Returns:
            dict:
                A mapping of table name to synthesizers.
        """
        self.metadata = metadata
        synthesizers = {
            table_name: self._get_trained_synthesizer(table, metadata.tables[table_name])
            for table_name, table in data.items()
        }
        self.table_columns = {table_name: data[table_name].columns for table_name in data.keys()}

        return synthesizers

    def _get_foreign_keys(self, metadata, table_name, child_name):
        foreign_keys = []
        for relation in metadata.relationships:
            if table_name == relation['parent_table_name'] and \
               child_name == relation['child_table_name']:
                foreign_keys.append(relation['child_foreign_key'])

        return foreign_keys

    def sample_from_synthesizer(self, synthesizers, n_samples):
        """Sample from the given synthesizers.

        Args:
            synthesizers (dict):
                A dict mapping table name to table synthesizer.
            n_samples (int):
                The number of samples.

        Returns:
            dict:
                A mapping of table name to sampled table data.
        """
        tables = {
            table_name: self._sample_from_synthesizer(synthesizer, n_samples)
            for table_name, synthesizer in synthesizers.items()
        }

        for table_name, table in tables.items():
            parents = list(self.metadata.tables[table_name]._get_parent_map().keys())
            for parent_name in parents:
                parent = tables[parent_name]
                primary_key = self.metadata.tables[table_name].primary_key
                foreign_keys = self._get_foreign_keys(self.metadata, parent_name, table_name)
                for foreign_key in foreign_keys:
                    foreign_key_values = parent[primary_key].sample(len(table), replace=True)
                    table[foreign_key] = foreign_key_values.values

            tables[table_name] = table[self.table_columns[table_name]]

        return tables
