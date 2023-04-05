"""Metrics module."""

import sdmetrics


class WithKWargs:

    def __init__(self, metric, **kwargs):
        self._metric = metric
        self._kwargs = kwargs

    def compute(self, real_data, synthetic_data, metadata):
        return self._metric.compute(real_data, synthetic_data, metadata, **self._kwargs)

    def normalize(self, raw_score):
        return self._metric.normalize(raw_score)


# Metrics to use by default for specific problem types and data
# modalities if no metrics have been explicitly specified.
PROBLEM_TYPE_METRICS = {
    'binary_classification': [
        'BinaryDecisionTreeClassifier',
        'BinaryAdaBoostClassifier',
        'BinaryLogisticRegression',
        'BinaryMLPClassifier',
    ],
    'multiclass_classification': [
        'MulticlassDecisionTreeClassifier',
        'MulticlassMLPClassifier',
    ],
    'regression': [
        'LinearRegression',
        'MLPRegressor',
    ],
    'bayesian_likelihood': [
        'BNLogLikelihood',
    ],
    'gaussian_likelihood': [
        (
            'GMLogLikelihood(10)',
            WithKWargs(sdmetrics.single_table.GMLogLikelihood, n_components=10, iterations=10),
        ),
        (
            'GMLogLikelihood(30)',
            WithKWargs(sdmetrics.single_table.GMLogLikelihood, n_components=30, iterations=10),
        ),
    ],
}
DATA_MODALITY_METRICS = {
    'single-table': [
        'CSTest',
        'KSComplement',
    ],
    'multi-table': [
        'CSTest',
        'KSComplement',
    ],
    'timeseries': [
        'TSFClassifierEfficacy',
        'LSTMClassifierEfficacy',
        'TSFCDetection',
        'LSTMDetection',
    ],
}


def get_metrics(metrics, metadata, modality):
    if modality == 'multi-table':
        metric_classes = sdmetrics.multi_table.MultiTableMetric.get_subclasses()
    elif modality == 'single-table':
        metric_classes = sdmetrics.single_table.SingleTableMetric.get_subclasses()
    elif modality == 'timeseries':
        metric_classes = sdmetrics.timeseries.TimeSeriesMetric.get_subclasses()

    if not metrics:
        metrics = DATA_MODALITY_METRICS[modality]

    final_metrics = {}
    metric_kwargs = {}
    for metric in metrics:
        metric_args = None
        if isinstance(metric, tuple):
            metric, metric_args = metric
        if isinstance(metric, str):
            metric_name = metric
            try:
                metric = metric_classes[metric]
            except KeyError:
                raise ValueError(f'Unknown {modality} metric: {metric}') from None

        else:
            metric_name = metric.__name__

        final_metrics[metric_name] = metric
        if metric_args:
            metric_kwargs[metric_name] = metric_args

    return final_metrics, metric_kwargs
