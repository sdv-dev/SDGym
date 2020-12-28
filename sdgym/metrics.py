"""Metrics module."""

import sdmetrics


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
        'GMLogLikelihood',
    ],
}


def get_metrics(metrics, metadata):
    modality = metadata._metadata['modality']

    if modality == 'multi-table':
        metric_classes = sdmetrics.multi_table.MultiTableMetric.get_subclasses()
    elif modality == 'single-table':
        metric_classes = sdmetrics.single_table.SingleTableMetric.get_subclasses()
    elif modality == 'timeseries':
        metric_classes = sdmetrics.timeseres.TimeSeriesMetric.get_subclasses()

    if not metrics:
        problem_type = metadata._metadata.get('problem_type')
        if problem_type:
            metrics = PROBLEM_TYPE_METRICS[problem_type]
        else:
            # TODO: Define specific default subset of metrics by data modality
            # instead of using all the metrics we have.
            return metric_classes

    final_metrics = {}
    for metric in metrics:
        if isinstance(metric, str):
            try:
                final_metrics[metric] = metric_classes[metric]
            except KeyError:
                raise ValueError(f'Unknown {modality} metric: {metric}')

    return final_metrics
