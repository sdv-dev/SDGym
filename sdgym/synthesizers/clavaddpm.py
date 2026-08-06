"""ClavaDDPMSynthesizer -- SDGym synthesizer for ClavaDDPM.

Paper: "ClavaDDPM: Multi-relational Data Synthesis with Cluster-guided Diffusion Models (2024)"
https://arxiv.org/abs/2405.17724

Original implementation is provided:
https://github.com/weipang142857/ClavaDDPM/tree/main.
"""

import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import sklearn
import torch
from packaging.version import Version
from sdv.metadata import Metadata
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from sdgym.synthesizers.base import MultiTableBaselineSynthesizer
from sdgym.synthesizers.tabddpm import CAT_MISSING_VALUE, TabDDPM, ohe_to_categories

SKLEARN_VERSION = Version(sklearn.__version__)


def guess_array_datetime_format(values, sample_size=100, dayfirst=False):
    """Guess the most likely datetime format via majority vote over a sample.

    Args:
        values (list, numpy.array, pd.Series):
            List of datetime values to inspect.
        sample_size (int, optional):
            Number of samples to use for guessing. Default 100.
        dayfirst (bool, optional):
            If True parses dates with the day first. Default False.

    Returns:
        str: Datetime format string, or None if it can not be guessed.
    """
    pandas_version = Version(pd.__version__).release[:2]
    if pandas_version >= (2, 2):
        from pandas.tseries.api import guess_datetime_format  # pandas >= 2.2.0
    else:
        from pandas._libs.tslibs.parsing import guess_datetime_format  # pandas < 2.2.0

    series = pd.Series(values)
    series = series.dropna()
    if series.empty:
        return None

    sample_size = min(sample_size, len(series))
    sample = series.sample(sample_size, random_state=0)
    sample = sample.astype(str).str.strip()
    sample = sample[sample != '']

    guesses = sample.apply(guess_datetime_format, dayfirst=dayfirst).dropna()
    counts = Counter(guesses)
    return top[0][0] if (top := counts.most_common(1)) else None


def get_group_data_dict(np_data, group_id_attrs=[0]):
    """Grouping dictionary from pipeline_utils.py."""
    group_data_dict = {}
    data_len = len(np_data)
    for i in range(data_len):
        row_id = tuple(np_data[i, group_id_attrs])
        if row_id not in group_data_dict:
            group_data_dict[row_id] = []
        group_data_dict[row_id].append(np_data[i])
    return group_data_dict


def get_group_data(np_data, group_id_attrs=[0]):
    """Grouping list from pipeline_utils.py."""
    group_data_list = []
    data_len = len(np_data)
    i = 0
    while i < data_len:
        group = []
        row_id = np_data[i, group_id_attrs]
        while (np_data[i, group_id_attrs] == row_id).all():
            group.append(np_data[i])
            i += 1
            if i >= data_len:
                break
        group_data_list.append(np.array(group))
    return np.array(group_data_list, dtype=object)


def min_max_normalize_sklearn(matrix):
    """Apply MinMaxScaler to each column from pipeline_utils.py."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_data = np.empty((matrix.shape[0], 0))
    for col in range(matrix.shape[1]):
        column = matrix[:, col].reshape(-1, 1)
        transformed_column = scaler.fit_transform(column)
        normalized_data = np.concatenate((normalized_data, transformed_column), axis=1)

    return normalized_data


def aggregate_and_sample(cluster_probabilities, child_group_lengths):
    """Aggregate the distribution and sample from pipeline_modules.py."""
    group_cluster_labels = []
    curr_index = 0
    agree_rates = []
    for group_length in child_group_lengths:
        group_probability_distribution = np.mean(
            cluster_probabilities[curr_index : curr_index + group_length], axis=0
        )
        group_cluster_label = np.random.choice(
            range(len(group_probability_distribution)), p=group_probability_distribution
        )
        group_cluster_labels.append(group_cluster_label)
        agree_rates.append(np.max(group_probability_distribution))
        curr_index += group_length
    return group_cluster_labels, agree_rates


def freq_to_prob(freq_dict):
    """Converts a dict of frequencies to a dict of probabilities from pipeline_utils.py."""
    prob_dict = {}
    for key in freq_dict:
        prob_dict[key] = freq_dict[key] / sum(list(freq_dict.values()))
    return prob_dict


def sample_from_dict(probabilities):
    """Sample using a dict of probabilities from pipeline_utils.py."""
    random_number = random.random()
    cumulative_sum = 0
    selected_key = None
    for key, probability in probabilities.items():
        cumulative_sum += probability
        if cumulative_sum >= random_number:
            selected_key = key
            break
    return selected_key


def get_df_without_id(df, id_cols):
    """Drop id columns based on `id_cols` from pipeline_utils.py."""
    return df.drop(columns=[col for col in id_cols if col in df.columns])


def calculate_days_since_earliest_date(dates, date_format='%y%m%d'):
    """Encode a date column as integer days since its earliest date from preprocess_utils.py."""
    parsed = [
        datetime.strptime(str(date), date_format) if pd.notna(date) else None for date in dates
    ]
    earliest_date = min(date for date in parsed if date is not None)
    days_since = [(date - earliest_date).days if date is not None else np.nan for date in parsed]
    return days_since, earliest_date.strftime(date_format)


def reconstruct_dates(days_since, earliest_date_str, date_format='%y%m%d'):
    """Inverse of `calculate_days_since_earliest_date` from preprocess_utils.py."""
    earliest_date = datetime.strptime(earliest_date_str, date_format)
    return [
        (earliest_date + timedelta(days=int(round(float(days))))).strftime(date_format)
        if pd.notna(days)
        else None
        for days in days_since
    ]


def table_label_encode(df, discrete_cols):
    """Label-encode the discrete columns of a table from preprocess_utils.py."""
    df = df.copy()
    label_encoders = {}
    for col in discrete_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders


def table_label_decode(df, label_encoders):
    """Inverse of :func:`table_label_encode` from preprocess_utils.py."""
    df = df.copy()
    for col, le in label_encoders.items():
        df[col] = le.inverse_transform(df[col])
    return df


def get_domain(df, id_cols, discrete_cols):
    """Build the ``{col: {'size', 'type'}}`` domain of a table from preprocess_utils.py."""
    domain = {}
    for col in df.columns:
        if col in discrete_cols:
            domain[col] = {'size': len(df[col].unique()), 'type': 'discrete'}
        elif col not in id_cols:
            domain[col] = {'size': len(df[col].unique()), 'type': 'continuous'}
    return domain


def topological_sort(graph):
    """Order tables into ``[parent, child]`` relations from preprocess_utils.py."""
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for child in graph[node]['children']:
            in_degree[child] += 1

    zero_in_degree = [node for node, degree in in_degree.items() if degree == 0]

    sorted_order = []
    for node in zero_in_degree:
        sorted_order.append([None, node])

    queue = zero_in_degree[:]
    while queue:
        current = queue.pop(0)
        for child in graph[current]['children']:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
            sorted_order.append([current, child])

    return sorted_order


def pair_clustering_keep_id(
    child_df,
    child_domain_dict,
    parent_df,
    parent_domain_dict,
    child_primary_key,
    parent_primary_key,
    foreign_key,
    num_clusters,
    parent_scale,
    key_scale,
    parent_name,
    child_name,
    clustering_method='kmeans',
    seed=0,
):
    """Cluster child rows augmented with their parent's features from pipeline_modules.py."""
    original_child_cols = list(child_df.columns)
    original_parent_cols = list(parent_df.columns)

    relation_cluster_name = f'{parent_name}_{child_name}_cluster'

    child_data = child_df.to_numpy()
    parent_data = parent_df.to_numpy()

    child_num_cols = []
    child_cat_cols = []

    parent_num_cols = []
    parent_cat_cols = []

    for col_index, col in enumerate(original_child_cols):
        if col in child_domain_dict:
            if child_domain_dict[col]['type'] == 'discrete':
                child_cat_cols.append((col_index, col))
            else:
                child_num_cols.append((col_index, col))

    for col_index, col in enumerate(original_parent_cols):
        if col in parent_domain_dict:
            if parent_domain_dict[col]['type'] == 'discrete':
                parent_cat_cols.append((col_index, col))
            else:
                parent_num_cols.append((col_index, col))

    child_primary_key_index = original_child_cols.index(child_primary_key)
    parent_primary_key_index = original_parent_cols.index(parent_primary_key)
    foreing_key_index = original_child_cols.index(foreign_key)

    # sort child data by foreign key
    sorted_child_data = child_data[np.argsort(child_data[:, foreing_key_index])]
    child_group_data_dict = get_group_data_dict(sorted_child_data, [foreing_key_index])

    # sort parent data by primary key
    sorted_parent_data = parent_data[np.argsort(parent_data[:, parent_primary_key_index])]

    group_lengths = []
    unique_group_ids = sorted_parent_data[:, parent_primary_key_index]
    for group_id in unique_group_ids:
        group_id = tuple([group_id])
        if group_id not in child_group_data_dict:
            group_lengths.append(0)
        else:
            group_lengths.append(len(child_group_data_dict[group_id]))

    group_lengths = np.array(group_lengths, dtype=int)

    sorted_parent_data_repeated = np.repeat(sorted_parent_data, group_lengths, axis=0)
    assert (
        sorted_parent_data_repeated[:, parent_primary_key_index]
        == sorted_child_data[:, foreing_key_index]
    ).all()

    child_group_data = get_group_data(sorted_child_data, [foreing_key_index])

    sorted_child_num_data = sorted_child_data[:, [ci for ci, _ in child_num_cols]]
    sorted_child_cat_data = sorted_child_data[:, [ci for ci, _ in child_cat_cols]]
    sorted_parent_num_data = sorted_parent_data_repeated[:, [ci for ci, _ in parent_num_cols]]
    sorted_parent_cat_data = sorted_parent_data_repeated[:, [ci for ci, _ in parent_cat_cols]]

    joint_num_matrix = np.concatenate([sorted_child_num_data, sorted_parent_num_data], axis=1)
    joint_cat_matrix = np.concatenate([sorted_child_cat_data, sorted_parent_cat_data], axis=1)

    # Impute missing numerical values with their column mean before clustering.
    # The reference assumed pre-imputed data; this mirrors the mean-fill that
    # TabDDPM's own transformer applies, so clustering (which cannot take NaNs)
    # is robust to real-world tables with missing values.
    if joint_num_matrix.shape[1] > 0:
        joint_num_matrix = joint_num_matrix.astype(float)
        col_means = np.nanmean(joint_num_matrix, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_positions = np.where(np.isnan(joint_num_matrix))
        joint_num_matrix[nan_positions] = np.take(col_means, nan_positions[1])

    joint_num_matrix_p_index = sorted_child_num_data.shape[1]
    cat_one_hot = None
    if joint_cat_matrix.shape[1] > 0:
        joint_cat_matrix_p_index = sorted_child_cat_data.shape[1]

        cat_converted = []
        for i in range(joint_cat_matrix.shape[1]):
            # skip huge categoricals to avoid an explosive one-hot encoding
            if len(np.unique(joint_cat_matrix[:, i])) > 1000:
                continue
            label_encoder = LabelEncoder()
            cat_converted.append(label_encoder.fit_transform(joint_cat_matrix[:, i]).astype(float))
        cat_converted = np.vstack(cat_converted).T

        cat_one_hot = np.empty((cat_converted.shape[0], 0))
        for col in range(cat_converted.shape[1]):
            if SKLEARN_VERSION.release[:2] >= (1, 2):
                encoder = OneHotEncoder(sparse_output=False)
            else:
                encoder = OneHotEncoder(sparse=False)

            column = cat_converted[:, col].reshape(-1, 1)
            cat_one_hot = np.concatenate((cat_one_hot, encoder.fit_transform(column)), axis=1)

        cat_one_hot[:, joint_cat_matrix_p_index:] = (
            parent_scale * cat_one_hot[:, joint_cat_matrix_p_index:]
        )

    num_min_max = min_max_normalize_sklearn(joint_num_matrix)

    # key channel: parent identity, factorized so arbitrary key types normalize
    key_factorized = pd.factorize(sorted_parent_data_repeated[:, parent_primary_key_index])[0]
    key_min_max = min_max_normalize_sklearn(key_factorized.astype(float).reshape(-1, 1))
    key_scaled = key_scale * key_min_max

    num_min_max[:, joint_num_matrix_p_index:] = (
        parent_scale * num_min_max[:, joint_num_matrix_p_index:]
    )

    if joint_cat_matrix.shape[1] > 0:
        cluster_data = np.concatenate((num_min_max, cat_one_hot, key_scaled), axis=1)
    else:
        cluster_data = np.concatenate((num_min_max, key_scaled), axis=1)

    child_group_lengths = np.array([len(group) for group in child_group_data], dtype=int)
    num_clusters = min(num_clusters, len(cluster_data))

    init_param = 'k-means++'
    if SKLEARN_VERSION.release[:2] < (1, 1):
        init_param = 'kmeans'

    if clustering_method == 'kmeans':
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto', init='k-means++', random_state=seed)
        kmeans.fit(cluster_data)
        cluster_labels = kmeans.labels_
    elif clustering_method == 'both':
        gmm = GaussianMixture(
            n_components=num_clusters,
            covariance_type='diag',
            init_params=init_param,
            tol=0.0001,
            random_state=seed,
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict(cluster_data)
    elif clustering_method == 'variational':
        gmm = BayesianGaussianMixture(
            n_components=num_clusters,
            covariance_type='diag',
            init_params=init_param,
            tol=0.0001,
            random_state=seed,
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict_proba(cluster_data)
    elif clustering_method == 'gmm':
        gmm = GaussianMixture(n_components=num_clusters, covariance_type='diag', random_state=seed)
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict(cluster_data)

    if clustering_method == 'variational':
        group_cluster_labels, agree_rates = aggregate_and_sample(
            cluster_labels, child_group_lengths
        )
    else:
        # voting to determine the cluster label for each parent
        group_cluster_labels = []
        curr_index = 0
        agree_rates = []
        for group_length in child_group_lengths:
            most_common_label_count = np.max(
                np.bincount(cluster_labels[curr_index : curr_index + group_length])
            )
            group_cluster_label = np.argmax(
                np.bincount(cluster_labels[curr_index : curr_index + group_length])
            )
            group_cluster_labels.append(group_cluster_label)
            agree_rates.append(most_common_label_count / group_length)
            curr_index += group_length

    group_assignment = np.repeat(group_cluster_labels, child_group_lengths, axis=0).reshape((-1, 1))
    sorted_child_data_with_cluster = np.concatenate([sorted_child_data, group_assignment], axis=1)

    # per-cluster distribution of "how many children a parent has"
    group_labels_list = group_cluster_labels
    group_lengths_list = child_group_lengths.tolist()
    group_lengths_dict = {}
    for i in range(len(group_labels_list)):
        group_label = group_labels_list[i]
        if group_label not in group_lengths_dict:
            group_lengths_dict[group_label] = defaultdict(int)
        group_lengths_dict[group_label][group_lengths_list[i]] += 1

    group_lengths_prob_dicts = {
        group_label: freq_to_prob(freq_dict)
        for group_label, freq_dict in group_lengths_dict.items()
    }

    # attach cluster label to the child in its original order
    sorted_child_ids = sorted_child_data[:, child_primary_key_index]
    child_id_to_cluster = dict(zip(sorted_child_ids, group_assignment.flatten()))
    child_df_with_cluster = child_df.copy()
    child_df_with_cluster[relation_cluster_name] = (
        child_df[child_primary_key].map(child_id_to_cluster).astype(int)
    )

    # a parent inherits its children's cluster; childless parents get a fresh id
    parent_id_to_cluster = {}
    for i in range(len(sorted_child_data)):
        parent_id = sorted_child_data[i, foreing_key_index]
        if parent_id in parent_id_to_cluster:
            continue
        parent_id_to_cluster[parent_id] = sorted_child_data_with_cluster[i, -1]

    max_cluster_label = max(parent_id_to_cluster.values())
    parent_df_with_cluster = parent_df.copy()
    parent_clusters = parent_df[parent_primary_key].map(parent_id_to_cluster)
    parent_df_with_cluster[relation_cluster_name] = parent_clusters.fillna(
        max_cluster_label + 1
    ).astype(int)

    new_col_entry = {
        'type': 'discrete',
        'size': len(set(parent_df_with_cluster[relation_cluster_name])),
    }
    parent_domain_dict[relation_cluster_name] = new_col_entry.copy()
    child_domain_dict[relation_cluster_name] = new_col_entry.copy()

    return parent_df_with_cluster, child_df_with_cluster, group_lengths_prob_dicts


def match_tables(A, B, n_clusters=25, unique_matching=True, batch_size=100):
    """Nearest-neighbour match of every row of ``A`` to a row of ``B``.

    n_clusters is used by the original implementation for faiss.IndexIVFFlat.
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    if not unique_matching:
        distances, indices = NearestNeighbors(n_neighbors=1).fit(B).kneighbors(A)
        return indices.flatten().tolist(), distances.flatten().tolist()

    k = min(len(B), 50)
    distances, indices = NearestNeighbors(n_neighbors=k).fit(B).kneighbors(A)
    used = set()
    matched_indices = []
    matched_distances = []
    for row_indices, row_distances in zip(indices, distances):
        chosen, chosen_distance = None, 0.0
        for candidate, distance in zip(row_indices, row_distances):
            if int(candidate) not in used:
                chosen, chosen_distance = int(candidate), float(distance)
                break
        if chosen is None:
            chosen = next(j for j in range(len(B)) if j not in used)
        used.add(chosen)
        matched_indices.append(chosen)
        matched_distances.append(chosen_distance)

    return matched_indices, matched_distances


def handle_multi_parent(
    child,
    parents,
    synthetic_tables,
    id_cols,
    n_clusters=1,
    unique_matching=True,
    batch_size=100,
    no_matching=False,
):
    """Reconcile a child generated once per parent into a single table."""
    synthetic_child_dfs = [
        (synthetic_tables[(parent, child)]['df'].copy(), fk) for parent, fk, _ppk in parents
    ]
    anchor_index = int(np.argmin([len(df) for df, _ in synthetic_child_dfs]))
    anchor = synthetic_child_dfs[anchor_index]
    synthetic_child_dfs.pop(anchor_index)
    for df, fk in synthetic_child_dfs:
        df_without_ids = get_df_without_id(df, id_cols)
        anchor_df_without_ids = get_df_without_id(anchor[0], id_cols)

        df_val = df_without_ids.to_numpy().astype(float)
        anchor_val = anchor_df_without_ids.to_numpy().astype(float)
        if len(df_val.shape) == 1:
            df_val = df_val.reshape(-1, 1)
            anchor_val = anchor_val.reshape(-1, 1)

        indices, _ = match_tables(
            anchor_val,
            df_val,
            n_clusters=n_clusters,
            unique_matching=unique_matching,
            batch_size=batch_size,
        )
        if no_matching:
            indices = np.random.permutation(indices)
        df = df.iloc[indices]
        anchor[0][fk] = df[fk].to_numpy()
    return anchor[0]


@torch.no_grad()
def _sample_step(diffusion, y):
    """One reverse-diffusion pass conditioned on the label tensor ``y``.

    Mirrors ``GaussianMultinomialDiffusion._sample`` but takes explicit
    per-row labels instead of drawing them from the empirical distribution.
    Similar to the original classifier-guided ``conditional_sample`` implemtation.
    """
    device = diffusion.log_alpha.device
    b = y.shape[0]
    z_norm = torch.randn((b, diffusion.num_numerical_features), device=device)

    has_cat = diffusion.num_classes[0] != 0
    log_z = torch.zeros((b, 0), device=device).float()
    if has_cat:
        uniform_logits = torch.zeros((b, len(diffusion.num_classes_expanded)), device=device)
        log_z = diffusion._log_sample_categorical(uniform_logits)

    out_dict = {'y': y.long().to(device)}
    for i in reversed(range(diffusion.num_timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        model_out = diffusion._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict)
        model_out_num = model_out[:, : diffusion.num_numerical_features]
        model_out_cat = model_out[:, diffusion.num_numerical_features :]
        if diffusion.num_numerical_features > 0:
            z_norm = diffusion._gaussian_p_sample(model_out_num, z_norm, t)['sample']
        if has_cat:
            log_z = diffusion._p_sample(model_out_cat, log_z, t)

    z_cat = log_z
    if has_cat:
        z_cat = ohe_to_categories(torch.exp(log_z).round(), diffusion.num_classes)
    return torch.cat([z_norm, z_cat], dim=1).cpu()


def _batched_conditional_sample(diffusion, y_codes, batch_size):
    """Generate one row per entry of ``y_codes``, retrying NaN rows."""
    n = len(y_codes)
    n_num = diffusion.num_numerical_features
    n_cat = len(diffusion.num_classes) if diffusion.num_classes[0] != 0 else 0
    out = np.zeros((n, n_num + n_cat), dtype='float32')

    pending = np.arange(n)
    for _ in range(20):
        if len(pending) == 0:
            break
        still = []
        for start in range(0, len(pending), batch_size):
            idx = pending[start : start + batch_size]
            sample = _sample_step(diffusion, torch.from_numpy(y_codes[idx])).numpy()
            bad = np.isnan(sample).any(axis=1)
            out[idx[~bad]] = sample[~bad]
            still.extend(idx[bad].tolist())
        pending = np.array(still, dtype=int)

    return out


def sample_from_diffusion(model, labels, sample_batch_size):
    """Decode a fitted ``TabDDPM`` conditioned on cluster ``labels``."""
    if len(labels) == 0:
        return pd.DataFrame(columns=list(model._column_order))

    y_str = np.array([str(int(v)) for v in labels]).reshape(-1, 1)
    y_codes = model._target_encoder.transform(y_str).reshape(-1).astype('int64')

    diffusion = model._diffusion
    diffusion.eval()
    x_gen = _batched_conditional_sample(diffusion, y_codes, sample_batch_size)

    n_num = diffusion.num_numerical_features
    has_cat = model._transformer.category_sizes[0] != 0
    x_num = x_gen[:, :n_num]
    x_cat = x_gen[:, n_num:] if has_cat else np.empty((len(x_gen), 0), dtype='int64')
    df = model._transformer.inverse_transform(x_num, x_cat)

    decoded = model._target_encoder.inverse_transform(y_codes.reshape(-1, 1)).reshape(-1)
    series = pd.Series(decoded, dtype=object).where(lambda s: s != CAT_MISSING_VALUE, np.nan)
    if model._target_is_boolean:
        series = series.map({'True': True, 'False': False})
    else:
        try:
            series = series.astype(model._target_dtype)
        except (ValueError, TypeError):
            pass
    df[model.target_column] = series

    return df[[column for column in model._column_order if column in df.columns]]


class ClavaDDPM:
    """Cluster-guided multi-table diffusion synthesizer.

    Args:
        metadata (sdv.metadata.Metadata):
            The metadata describing the data.
        num_clusters (int or dict):
            Latent clusters per relationship; an int applies to all, a dict maps
            a child table name to its own count.
        parent_scale (float):
            Weight of the parent's features when clustering.
        key_scale (float):
            Weight of the parent-identity (key) channel when clustering.
        clustering_method (str):
            ``'both'`` / ``'gmm'`` (Gaussian mixture), ``'kmeans'`` or
            ``'variational'`` (Bayesian GMM).
        d_layers (List[int]):
            Hidden layer sizes of the MLP denoiser.
        dropout (float):
            Dropout of the MLP denoiser.
        dim_t (int):
            Timestep/label embedding dimension.
        num_timesteps (int):
            Diffusion timesteps T.
        scheduler (str):
            ``'cosine'`` or ``'linear'`` beta schedule.
        diffusion_iterations (int):
            Training iterations.
        batch_size (int):
            Batch size.
        lr (float):
            Learning rate for the optimizer.
        weight_decay (float):
            Weight decay for the optimizer.
        sampling_batch_size (int):
            Batch size used during sampling.
        num_matching_clusters (int):
            Number of clusters to match. Default 1.
        matching_batch_size (int):
            Number of samples in the batch. Default 1000.
        unique_matching (bool):
            Multi-parent matching controls for unique matching.
        no_matching (bool):
            Multi-parent matching controls for no matching.
        max_categories (int):
            Maximum number of distinct values kept for a categorical column.
            Columns with more are reduced to this many values, randomly sampled
            from the data.
        device (str or None):
            Whether to use ``'cuda'`` or ``'cpu'``. If None, auto-select is used.
        seed (int):
            Random seed.
        verbose (bool):
            Show progress.
    """

    def __init__(
        self,
        metadata,
        num_clusters=25,
        parent_scale=1.0,
        key_scale=1.0,
        clustering_method='both',
        d_layers=(512, 1024, 1024, 1024, 1024, 512),
        dropout=0.0,
        dim_t=128,
        num_timesteps=2000,
        scheduler='cosine',
        diffusion_iterations=100000,
        batch_size=4096,
        lr=6e-4,
        weight_decay=1e-5,
        sampling_batch_size=20000,
        num_matching_clusters=1,
        matching_batch_size=1000,
        unique_matching=True,
        no_matching=False,
        max_categories=100,
        device=None,
        seed=0,
        verbose=False,
    ):
        self.num_clusters = num_clusters
        self.parent_scale = parent_scale
        self.key_scale = key_scale
        self.clustering_method = clustering_method
        self.d_layers = list(d_layers)
        self.dropout = dropout
        self.dim_t = dim_t
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.diffusion_iterations = diffusion_iterations
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.sampling_batch_size = sampling_batch_size
        self.num_matching_clusters = num_matching_clusters
        self.matching_batch_size = matching_batch_size
        self.unique_matching = unique_matching
        self.no_matching = no_matching
        self.max_categories = max_categories
        self.device = device
        self.seed = seed
        self.verbose = verbose

        self._parse_metadata(metadata)
        self._fitted = False

    def _parse_metadata(self, metadata):
        meta = metadata if isinstance(metadata, dict) else metadata.to_dict()

        self._table_names = list(meta['tables'])
        self._primary_key = {}
        # child -> list of (parent_table, foreign_key_col, parent_primary_key)
        self._parents = {name: [] for name in self._table_names}
        self._children = {name: [] for name in self._table_names}

        for name, table_meta in meta['tables'].items():
            self._primary_key[name] = table_meta.get('primary_key')

        for relationship in meta.get('relationships', []):
            parent = relationship['parent_table_name']
            child = relationship['child_table_name']
            self._parents[child].append((
                parent,
                relationship['child_foreign_key'],
                relationship['parent_primary_key'],
            ))
            self._children[parent].append(child)

        # Resolve a primary key for every table.
        self._primary_key = {}
        self._synthetic_pk = set()
        for name, table_meta in meta['tables'].items():
            pk = table_meta.get('primary_key')
            if pk is None:
                pk = f'__{name}_pk__'
                self._synthetic_pk.add(name)

            self._primary_key[name] = pk

        # Preprocessing
        self._discrete_cols = {}
        self._datetime_cols = {}  # name -> {col: datetime_format}
        for name, table_meta in meta['tables'].items():
            id_cols = self._id_cols(name)
            discrete, datetimes = [], {}
            for column, spec in table_meta['columns'].items():
                sdtype = spec.get('sdtype', 'categorical')
                if column in id_cols:
                    continue
                if sdtype == 'datetime':
                    datetimes[column] = spec.get('datetime_format')
                elif sdtype != 'numerical':
                    discrete.append(column)
            self._discrete_cols[name] = discrete
            self._datetime_cols[name] = datetimes

        graph = {name: {'children': self._children[name]} for name in self._table_names}
        self._relation_order = [tuple(edge) for edge in topological_sort(graph)]

    def _num_clusters_for(self, child):
        if isinstance(self.num_clusters, dict):
            return self.num_clusters[child]
        return self.num_clusters

    def _id_cols(self, table):
        return [self._primary_key[table]] + [fk for _p, fk, _ppk in self._parents[table]]

    def fit(self, data):
        """Fit this model to the original data.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame``.
        """
        missing = set(self._table_names) - set(data)
        if missing:
            raise ValueError(f'Missing tables in data: {sorted(missing)}')

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Preprocess every table up front: turn datetimes into numeric
        # day-counts, label-encode the discrete columns, and derive the
        # ``domain``. Encoders / date anchors are kept so ``sample`` can invert
        # them.
        self._tables = {}
        for name in self._table_names:
            output_cols = list(data[name].columns)
            df = data[name].copy().reset_index(drop=True)
            # tables without a primary key get an internal row-id
            if name in self._synthetic_pk:
                df[self._primary_key[name]] = np.arange(len(df))

            # cap the cardinality of the discrete columns.
            for col in self._discrete_cols[name]:
                uniques = df[col].dropna().unique()
                if len(uniques) > self.max_categories:
                    kept = np.random.choice(uniques, size=self.max_categories, replace=False)
                    outside = ~df[col].isin(kept)
                    df.loc[outside, col] = np.random.choice(kept, size=int(outside.sum()))

            date_info = {}
            for col, datetime_format in self._datetime_cols[name].items():
                if datetime_format is None:
                    datetime_array = df[df.notna()].astype(str).to_numpy()
                    datetime_format = guess_array_datetime_format(datetime_array)
                    df[col] = pd.to_datetime(
                        df[col], format=datetime_format, errors='coerce'
                    ).dt.strftime()

                days_since, earliest = calculate_days_since_earliest_date(df[col], datetime_format)
                df[col] = np.asarray(days_since, dtype=float)
                date_info[col] = (earliest, datetime_format)

            df, label_encoders = table_label_encode(df, self._discrete_cols[name])
            domain = get_domain(df, self._id_cols(name), self._discrete_cols[name])

            self._tables[name] = {
                'df': df,
                'domain': domain,
                'output_cols': output_cols,
                'original_len': len(data[name]),
                'pk': self._primary_key[name],
                'parents': list(self._parents[name]),
                'label_encoders': label_encoders,
                'date_info': date_info,
            }

        self._clustering()
        self._training()
        self._fitted = True

    def _clustering(self):
        """clava_clustering: cluster every parent->child relationship."""
        relation_order_reversed = self._relation_order[::-1]
        self._all_group_lengths_prob_dicts = {}

        for parent, child in relation_order_reversed:
            if parent is None:
                continue
            if self.verbose:
                sys.stdout.write(f'Clustering {parent} -> {child}\n')

            fk = next(f for p, f, _ppk in self._tables[child]['parents'] if p == parent)

            parent_df, child_df, group_lengths_prob_dicts = pair_clustering_keep_id(
                self._tables[child]['df'],
                self._tables[child]['domain'],
                self._tables[parent]['df'],
                self._tables[parent]['domain'],
                self._tables[child]['pk'],
                self._tables[parent]['pk'],
                fk,
                self._num_clusters_for(child),
                self.parent_scale,
                self.key_scale,
                parent,
                child,
                clustering_method=self.clustering_method,
                seed=self.seed,
            )
            self._tables[parent]['df'] = parent_df
            self._tables[child]['df'] = child_df
            self._all_group_lengths_prob_dicts[(parent, child)] = group_lengths_prob_dicts

    def _training(self):
        """clava_training: train one TabDDPM per relationship (child_training)."""
        self._models = {}
        for parent, child in self._relation_order:
            if self.verbose:
                sys.stdout.write(f'Training {parent} -> {child}\n')

            df_with_cluster = self._tables[child]['df']
            df_without_id = get_df_without_id(df_with_cluster, self._id_cols(child))
            self._models[(parent, child)] = self._child_training(
                df_without_id, self._tables[child]['domain'], parent, child
            )

    def _child_training(self, df_without_id, domain, parent, child):
        """child_training: fit a (conditional) TabDDPM for one relationship.

        Root relationships (``parent is None``) train unconditionally; child
        relationships condition on the ``{parent}_{child}_cluster`` column via
        TabDDPM's ``target_column`` label embedding.
        """
        y_col = None if parent is None else f'{parent}_{child}_cluster'

        columns_meta = {}
        for column in df_without_id.columns:
            if column not in domain:
                continue
            sdtype = 'categorical' if domain[column]['type'] == 'discrete' else 'numerical'
            columns_meta[column] = {'sdtype': sdtype}

        table_metadata = Metadata.load_from_dict({
            'tables': {child: {'columns': columns_meta}},
        })

        model = TabDDPM(
            table_metadata,
            target_column=y_col,
            d_layers=self.d_layers,
            dropout=self.dropout,
            dim_t=self.dim_t,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            steps=self.diffusion_iterations,
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            sample_batch_size=self.sampling_batch_size,
            device=self.device,
            seed=self.seed,
            verbose=self.verbose,
        )
        model.fit(df_without_id[list(columns_meta)])
        return model

    def sample(self, scale=1.0):
        """Generate synthetic data for the entire dataset.

        Args:
            scale (float):
                A float representing how much to scale the data by. If scale is set to ``1.0``,
                this does not scale the sizes of the tables. If ``scale`` is greater than ``1.0``
                create more rows than the original data by a factor of ``scale``.
                If ``scale`` is lower than ``1.0`` create fewer rows by the factor of ``scale``
                than the original tables. Defaults to ``1.0``.
        """
        if not self._fitted:
            raise RuntimeError('The synthesizer has not been fitted; call fit() first.')

        synthetic_tables = {}

        for parent, child in self._relation_order:
            if self.verbose:
                sys.stdout.write(f'Generating {parent} -> {child}\n')

            result = self._models[(parent, child)]
            df_with_cluster = self._tables[child]['df']
            df_without_id = get_df_without_id(df_with_cluster, self._id_cols(child))

            if parent is None:
                child_generated = result.sample(round(scale * len(df_without_id)))
                child_keys = list(range(len(child_generated)))
                generated = child_generated.copy()
                generated.insert(0, self._tables[child]['pk'], child_keys)
                synthetic_tables[(parent, child)] = {'df': generated, 'keys': child_keys}
            else:
                # any already-generated table for this parent carries its cluster
                for (_p, cname), value in synthetic_tables.items():
                    if cname == parent:
                        parent_synthetic_df = value['df']
                        parent_keys = value['keys']
                        break

                cluster_col = f'{parent}_{child}_cluster'
                group_labels = (
                    parent_synthetic_df[cluster_col].astype(float).round().astype(int).tolist()
                )
                group_lengths_prob_dicts = self._all_group_lengths_prob_dicts[(parent, child)]

                sampled_group_sizes = []
                ys = []
                for group_label in group_labels:
                    if group_label not in group_lengths_prob_dicts:
                        sampled_group_sizes.append(0)
                        continue
                    sampled_group_size = sample_from_dict(group_lengths_prob_dicts[group_label])
                    sampled_group_sizes.append(sampled_group_size)
                    ys.extend([group_label] * sampled_group_size)

                child_generated = sample_from_diffusion(
                    result, np.array(ys), self.sampling_batch_size
                )

                child_foreign_keys = np.repeat(parent_keys, sampled_group_sizes, axis=0)
                child_primary_keys = np.arange(len(child_generated))
                fk = next(f for p, f, _ppk in self._tables[child]['parents'] if p == parent)

                generated = child_generated.copy().reset_index(drop=True)
                generated[self._tables[child]['pk']] = child_primary_keys
                generated[fk] = child_foreign_keys
                synthetic_tables[(parent, child)] = {
                    'df': generated,
                    'keys': list(child_primary_keys),
                }

        final_tables = {}
        for parent, child in self._relation_order:
            if child in final_tables:
                continue
            if len(self._tables[child]['parents']) > 1:
                final_tables[child] = handle_multi_parent(
                    child,
                    self._tables[child]['parents'],
                    synthetic_tables,
                    self._id_cols(child),
                    n_clusters=self.num_matching_clusters,
                    unique_matching=self.unique_matching,
                    batch_size=self.matching_batch_size,
                    no_matching=self.no_matching,
                )
            else:
                final_tables[child] = synthetic_tables[(parent, child)]['df']

        return {child: self._postprocess(child, table) for child, table in final_tables.items()}

    def _postprocess(self, table, df):
        """Invert fit-time preprocessing: decode discretes, rebuild dates, order cols."""
        df = df.copy()
        state = self._tables[table]

        # decode label-encoded discrete columns back to their original values
        encoders = {col: le for col, le in state['label_encoders'].items() if col in df.columns}
        for col in encoders:
            df[col] = df[col].astype(int)
        df = table_label_decode(df, encoders)

        # turn numeric day-counts back into formatted date strings
        for col, (earliest, date_format) in state['date_info'].items():
            if col in df.columns:
                df[col] = reconstruct_dates(df[col].to_numpy(), earliest, date_format)

        return df[state['output_cols']].reset_index(drop=True)


class ClavaDDPMSynthesizer(MultiTableBaselineSynthesizer):
    """Custom wrapper for the ClavaDDPM synthesizer to make it work with SDGym."""

    LOGGER = logging.getLogger(__name__)
    _MODEL_KWARGS = None
    _MODALITY_FLAG = 'multi_table'

    def _fit(self, data, metadata):
        """Fit the synthesizer to the multi-table data.

        Args:
            data (dict):
                A dict mapping table name to table data.
            metadata (sdv.metadata.MultiTableMetadata):
                The multi-table metadata describing the data.
        """
        model_kwargs = self._MODEL_KWARGS.copy() if self._MODEL_KWARGS else {}
        model = ClavaDDPM(metadata, **model_kwargs)
        model.fit(data)

        self._internal_synthesizer = model

    def _sample_from_synthesizer(self, synthesizer, scale):
        """Sample data from the provided synthesizer.

        Args:
            synthesizer (SDGym synthesizer):
                The synthesizer object to sample data from.
            scale (float):
                The scale of data to sample.
                Defaults to 1.0.

        Returns:
            dict: A dict mapping table name to the sampled data.
        """
        return synthesizer._internal_synthesizer.sample(scale)
