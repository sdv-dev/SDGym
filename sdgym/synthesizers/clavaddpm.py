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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from tqdm import tqdm

from sdgym.synthesizers.base import MultiTableBaselineSynthesizer
from sdgym.synthesizers.tabddpm import (
    CAT_MISSING_VALUE,
    TabDDPM,
    _DataTransformer,
    ohe_to_categories,
)

SKLEARN_VERSION = Version(sklearn.__version__)


# --------------------------------------------------------------------------
# ######################## From preprocess_utils.py ########################
# --------------------------------------------------------------------------


def calculate_days_since_earliest_date(dates, date_format='%y%m%d'):
    """Encode a date column as integer days since its earliest date from preprocess_utils.py.

    Modified by SDGym to take different `date_format` and handle Nans.
    """
    parsed = [
        datetime.strptime(str(date), date_format) if pd.notna(date) else None for date in dates
    ]
    earliest_date = min(date for date in parsed if date is not None)
    days_since = [(date - earliest_date).days if date is not None else np.nan for date in parsed]
    return days_since, earliest_date.strftime(date_format)


def reconstruct_dates(days_since, earliest_date_str, date_format='%y%m%d'):
    """Inverse of `calculate_days_since_earliest_date` from preprocess_utils.py.

    Modified by SDGym to take different `date_format` and handle Nans.
    """
    earliest_date = datetime.strptime(earliest_date_str, date_format)
    original_dates = [
        (earliest_date + timedelta(days=int(round(float(days))))).strftime(date_format)
        if pd.notna(days)
        else None
        for days in days_since
    ]
    return original_dates


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
    """Inverse of `table_label_encode` from preprocess_utils.py."""
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
    # Initialize the indegree map and output
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for child in graph[node]['children']:
            in_degree[child] += 1

    # Queue for nodes with no incoming edges
    zero_in_degree = [node for node, degree in in_degree.items() if degree == 0]

    # Output list for storing the order
    sorted_order = []

    # Start with root nodes and format them with None as parent
    for node in zero_in_degree:
        sorted_order.append([None, node])

    # Using a queue to maintain nodes to process
    queue = zero_in_degree[:]

    while queue:
        current = queue.pop(0)
        for child in graph[current]['children']:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
            # Add each parent-child relationship as we process them
            sorted_order.append([current, child])

    return sorted_order


# --------------------------------------------------------------------------
# ######################### From pipeline_utils.py #########################
# --------------------------------------------------------------------------


def get_group_data_dict(
    np_data,
    group_id_attrs=[
        0,
    ],
):
    """Grouping dictionary from pipeline_utils.py."""
    group_data_dict = {}
    data_len = len(np_data)
    for i in range(data_len):
        row_id = tuple(np_data[i, group_id_attrs])
        if row_id not in group_data_dict:
            group_data_dict[row_id] = []
        group_data_dict[row_id].append(np_data[i])

    return group_data_dict


def get_group_data(
    np_data,
    group_id_attrs=[
        0,
    ],
):
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
        group = np.array(group)
        group_data_list.append(group)
    group_data_list = np.array(group_data_list, dtype=object)

    return group_data_list


def min_max_normalize_sklearn(matrix):
    """Apply MinMaxScaler to each column from pipeline_utils.py."""
    scaler = MinMaxScaler(feature_range=(-1, 1))

    normalized_data = np.empty((matrix.shape[0], 0))

    # Apply MinMaxScaler to each column and concatenate the results
    for col in range(matrix.shape[1]):
        column = matrix[:, col].reshape(-1, 1)
        transformed_column = scaler.fit_transform(column)
        normalized_data = np.concatenate((normalized_data, transformed_column), axis=1)

    return normalized_data


def freq_to_prob(freq_dict):
    """Converts a dict of frequencies to a dict of probabilities from pipeline_utils.py."""
    prob_dict = {}
    for key in freq_dict:
        prob_dict[key] = freq_dict[key] / sum(list(freq_dict.values()))
    return prob_dict


def sample_from_dict(probabilities):
    """Sample using a dict of probabilities from pipeline_utils.py."""
    # Generate a random number between 0 and 1
    random_number = random.random()

    # Initialize cumulative sum and the selected key
    cumulative_sum = 0
    selected_key = None

    # Iterate through the dictionary
    for key, probability in probabilities.items():
        cumulative_sum += probability
        if cumulative_sum >= random_number:
            selected_key = key
            break

    return selected_key


def get_df_without_id(df, id_cols):
    """Drop id columns based on `id_cols` from pipeline_utils.py."""
    id_cols = [col for col in df.columns if '_id' in col]
    return df.drop(columns=id_cols)


def convert_to_unique_indices(indices):
    """Convert indices to unique values from pipline_utils.py."""
    occurrence = set()
    max_index = len(indices)  # Assuming the range is the length of the list
    replacement_candidates = set(range(max_index)) - set(indices)

    for i, num in enumerate(tqdm(indices)):
        if num in occurrence:
            # Find the smallest number not in the list
            replacement = min(replacement_candidates)
            indices[i] = replacement
            replacement_candidates.remove(replacement)
        else:
            occurrence.add(num)

    return indices


def match_tables(A, B, n_clusters=25, unique_matching=True, batch_size=100):
    """Nearest-neighbour match of every row of ``A`` to a row of ``B`` from pipline_utils.py."""
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    # Dimension of vectors
    d = B.shape[1]

    if unique_matching:
        quantiser = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantiser, d, n_clusters, faiss.METRIC_L2)
    else:
        res = faiss.StandardGpuResources()
        quantiser = faiss.IndexFlatL2(d)
        index_cpu = faiss.IndexIVFFlat(quantiser, d, n_clusters, faiss.METRIC_L2)
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    index.train(B)
    index.add(B)

    # Initialize lists to store the results
    all_indices = []
    all_distances = []

    if unique_matching:
        batch_size = 1
        n_batches = (A.shape[0] + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, A.shape[0])
            D, I = index.search(A[start:end], k=1)  # noqa: E741
            index.remove_ids(I.flatten())
            all_distances.append(D)
            all_indices.append(I)

        # Concatenate the results from all batches
        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)
        distances = all_distances.flatten().tolist()
        indices = all_indices.flatten().tolist()
    else:
        n_batches = (A.shape[0] + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, A.shape[0])
            D, I = index.search(A[start:end], k=1)  # noqa: E741
            all_distances.append(D)
            all_indices.append(I)

        # Concatenate the results from all batches
        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)
        distances = all_distances.flatten().tolist()
        indices = all_indices.flatten().tolist()
        indices = convert_to_unique_indices(indices)
        assert len(indices) == len(set(indices))

    return indices, distances


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
    """Reconcile a child generated once per parent into a single table from pipeline_utils.py.

    Modified code to use 'fk' instead of assuming '{parent}_id'.
    """
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
            # randomly shuffle the array
            indices = np.random.permutation(indices)
        df = df.iloc[indices]
        anchor[0][fk] = df[fk].to_numpy()
    return anchor[0]


# --------------------------------------------------------------------------
# ######################## From pipeline_modules.py ########################
# --------------------------------------------------------------------------


def aggregate_and_sample(cluster_probabilities, child_group_lengths):
    """Aggregate the distribution and sample from pipeline_modules.py."""
    group_cluster_labels = []
    curr_index = 0
    agree_rates = []

    for group_length in child_group_lengths:
        # Aggregate the probability distributions by taking the mean
        group_probability_distribution = np.mean(
            cluster_probabilities[curr_index : curr_index + group_length], axis=0
        )

        # Sample the label from the aggregated distribution
        group_cluster_label = np.random.choice(
            range(len(group_probability_distribution)), p=group_probability_distribution
        )
        group_cluster_labels.append(group_cluster_label)

        # Compute the max probability as the agree rate
        max_probability = np.max(group_probability_distribution)
        agree_rates.append(max_probability)

        # Update the curr_index for the next iteration
        curr_index += group_length

    return group_cluster_labels, agree_rates


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
    """Cluster child rows augmented with their parent's features from pipeline_modules.py.

    Modified by SDGym to:
    * remove output statements
    * handle different versions of sklearn
    """
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

    parent_primary_key_index = original_parent_cols.index(parent_primary_key)
    foreing_key_index = original_child_cols.index(foreign_key)

    # sort child data by foreign key
    sorted_child_data = child_data[np.argsort(child_data[:, foreing_key_index])]
    child_group_data_dict = get_group_data_dict(
        sorted_child_data,
        [
            foreing_key_index,
        ],
    )

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

    if joint_cat_matrix.shape[1] > 0:
        joint_cat_matrix_p_index = sorted_child_cat_data.shape[1]
        joint_num_matrix_p_index = sorted_child_num_data.shape[1]

        cat_converted = []
        for i in range(joint_cat_matrix.shape[1]):
            # skip huge categoricals to avoid an explosive one-hot encoding
            if len(np.unique(joint_cat_matrix[:, i])) > 1000:
                continue
            label_encoder = LabelEncoder()
            cat_converted.append(label_encoder.fit_transform(joint_cat_matrix[:, i]).astype(float))
        cat_converted = np.vstack(cat_converted).T

        # Initialize an empty array to store the encoded values
        cat_one_hot = np.empty((cat_converted.shape[0], 0))

        # Loop through each column in the data and encode it
        for col in range(cat_converted.shape[1]):
            if SKLEARN_VERSION.release[:2] >= (1, 2):
                encoder = OneHotEncoder(sparse_output=False)
            else:
                encoder = OneHotEncoder(sparse=False)

            column = cat_converted[:, col].reshape(-1, 1)
            encoded_column = encoder.fit_transform(column)
            cat_one_hot = np.concatenate((cat_one_hot, encoded_column), axis=1)

        cat_one_hot[:, joint_cat_matrix_p_index:] = (
            parent_scale * cat_one_hot[:, joint_cat_matrix_p_index:]
        )

    num_min_max = min_max_normalize_sklearn(joint_num_matrix)

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

    n_init = 'auto'
    if SKLEARN_VERSION.release[:2] < (1, 2):
        n_init = 10

    if clustering_method == 'kmeans':
        kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, init='k-means++', random_state=seed)
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
            # First, determine the most common label in the current group
            most_common_label_count = np.max(
                np.bincount(cluster_labels[curr_index : curr_index + group_length])
            )
            group_cluster_label = np.argmax(
                np.bincount(cluster_labels[curr_index : curr_index + group_length])
            )
            group_cluster_labels.append(group_cluster_label)

            # Compute agree rate using the most common label count
            agree_rate = most_common_label_count / group_length
            agree_rates.append(agree_rate)

            # Then, update the curr_index for the next iteration
            curr_index += group_length

    group_assignment = np.repeat(group_cluster_labels, child_group_lengths, axis=0).reshape((-1, 1))
    sorted_child_data_with_cluster = np.concatenate([sorted_child_data, group_assignment], axis=1)

    group_labels_list = group_cluster_labels
    group_lengths_list = child_group_lengths.tolist()
    group_lengths_dict = {}
    for i in range(len(group_labels_list)):
        group_label = group_labels_list[i]
        if group_label not in group_lengths_dict:
            group_lengths_dict[group_label] = defaultdict(int)
        group_lengths_dict[group_label][group_lengths_list[i]] += 1

    group_lengths_prob_dicts = {}
    for group_label, freq_dict in group_lengths_dict.items():
        group_lengths_prob_dicts[group_label] = freq_to_prob(freq_dict)

    # recover the preprocessed data back to dataframe
    child_df_with_cluster = pd.DataFrame(
        sorted_child_data_with_cluster, columns=original_child_cols + [relation_cluster_name]
    )

    # recover child df order
    child_df_with_cluster = pd.merge(  # noqa: PD015
        child_df[[child_primary_key]],
        child_df_with_cluster,
        on=child_primary_key,
        how='left',
    )

    parent_id_to_cluster = {}
    for i in range(len(sorted_child_data)):
        parent_id = sorted_child_data[i, foreing_key_index]
        if parent_id in parent_id_to_cluster:
            assert parent_id_to_cluster[parent_id] == sorted_child_data_with_cluster[i, -1]
            continue
        parent_id_to_cluster[parent_id] = sorted_child_data_with_cluster[i, -1]

    max_cluster_label = max(parent_id_to_cluster.values())

    parent_data_clusters = []
    for i in range(len(parent_data)):
        if parent_data[i, parent_primary_key_index] in parent_id_to_cluster:
            parent_data_clusters.append(
                parent_id_to_cluster[parent_data[i, parent_primary_key_index]]
            )
        else:
            parent_data_clusters.append(max_cluster_label + 1)

    parent_data_clusters = np.array(parent_data_clusters).reshape(-1, 1)
    parent_data_with_cluster = np.concatenate([parent_data, parent_data_clusters], axis=1)
    parent_df_with_cluster = pd.DataFrame(
        parent_data_with_cluster, columns=original_parent_cols + [relation_cluster_name]
    )

    new_col_entry = {'type': 'discrete', 'size': len(set(parent_data_clusters.flatten()))}
    parent_domain_dict[relation_cluster_name] = new_col_entry.copy()
    child_domain_dict[relation_cluster_name] = new_col_entry.copy()

    return parent_df_with_cluster, child_df_with_cluster, group_lengths_prob_dicts


# --------------------------------------------------------------------------
# ########################### SDGym extra logic ############################
# --------------------------------------------------------------------------
def _check_faiss_installed():
    try:
        import faiss  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "In order to use 'ClavaDDPMSynthesizer' you have to install the extra "
            "dependencies by first installing 'faiss' library and other dependencies: \n\n"
            "    conda install -c pytorch -c nvidia faiss-gpu\n"
            "    pip install sdgym['clavaddpm']\n"
        ) from e


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


def decode_id_values(values, encoder, column):
    """Inverse-transform label-encoded id codes back to the original id values.

    Sampling can create more rows than the original table, so codes past the
    encoder's fitted range get fresh values: numeric ids continue after the
    largest original id, other ids become new ``'{column}_{code}'`` strings.

    Args:
        values (list, numpy.array, pd.Series):
            ID column values.
        encoder (sklearn.preprocessing.LabelEncoder):
            Fitted encoder.
        column (str):
            Column name.

    Returns:
        np.array:
            List of decoded id values.
    """
    codes = np.asarray(values).astype('int64')
    classes = encoder.classes_
    decoded = np.empty(len(codes), dtype=object)
    in_range = (codes >= 0) & (codes < len(classes))
    decoded[in_range] = encoder.inverse_transform(codes[in_range])
    if (~in_range).any():
        if pd.api.types.is_numeric_dtype(classes):
            # continue past the largest original id (code n_classes -> max + 1, ...)
            decoded[~in_range] = codes[~in_range] + (classes.max() + 1 - len(classes))
        else:
            decoded[~in_range] = [f'{column}_{code}' for code in codes[~in_range]]

    if pd.api.types.is_numeric_dtype(classes):
        return decoded.astype(classes.dtype)

    return decoded


@torch.no_grad()
def _sample_step(diffusion, y):
    """One reverse-diffusion pass conditioned on the label tensor ``y``.

    Mirrors ``GaussianMultinomialDiffusion._sample`` but takes explicit
    per-row labels instead of drawing them from the empirical distribution.
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
        device=None,
        seed=0,
        verbose=False,
    ):
        _check_faiss_installed()

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
        self.device = device
        self.seed = seed
        self.verbose = verbose

        self._parse_metadata(metadata)
        self._metadata = metadata
        self._fitted = False

    def _parse_metadata(self, metadata):
        meta = metadata if isinstance(metadata, dict) else metadata.to_dict()

        self._table_names = list(meta['tables'])
        # child -> list of (parent_table, foreign_key_col, parent_primary_key)
        self._parents = {name: [] for name in self._table_names}
        self._children = {name: [] for name in self._table_names}

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
            if pk is None or isinstance(pk, list):  # override composite key
                pk = f'__{name}_pk__'
                self._synthetic_pk.add(name)

            self._primary_key[name] = pk

        # Preprocessing
        self._discrete_cols = {}
        self._datetime_cols = {}  # name -> {col: datetime_format}
        self._id_value_cols = {}
        for name, table_meta in meta['tables'].items():
            id_cols = self._id_cols(name)
            discrete, datetimes, id_values = [], {}, []
            for column, spec in table_meta['columns'].items():
                sdtype = spec.get('sdtype', 'categorical')
                if column in id_cols:
                    continue
                if sdtype == 'id':
                    id_values.append(column)
                elif sdtype == 'datetime':
                    datetimes[column] = spec.get('datetime_format')
                elif sdtype != 'numerical':
                    discrete.append(column)
            self._discrete_cols[name] = discrete
            self._datetime_cols[name] = datetimes
            self._id_value_cols[name] = id_values

        graph = {name: {'children': self._children[name]} for name in self._table_names}
        self._relation_order = [tuple(edge) for edge in topological_sort(graph)]

    def _num_clusters_for(self, child):
        if isinstance(self.num_clusters, dict):
            return self.num_clusters[child]
        return self.num_clusters

    def _id_cols(self, table):
        return [self._primary_key[table]] + [fk for _p, fk, _ppk in self._parents[table]]

    def _get_values(self, data, table_name, key):
        values = [data[table_name][key].astype(str).dropna().to_numpy().flatten()]
        for child in self._table_names:
            for parent, fk, _ppk in self._parents[child]:
                if parent == table_name and fk in data[child].columns:
                    keys = data[child][fk].astype(str).dropna().to_numpy().flatten()
                    values.append(keys)

        return np.concatenate(values)

    def fit(self, data):
        """Fit this model to the original data.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame``.
        """
        self._metadata.validate_data(data)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self._id_encoded_cols = defaultdict(dict)
        for name in self._table_names:
            pk = self._primary_key[name]
            if not (
                pk is None
                or name in self._synthetic_pk
                or pd.api.types.is_numeric_dtype(data[name][pk])
            ):
                key_values = self._get_values(data, name, pk)
                encoder = LabelEncoder()
                encoder.fit(key_values)
                self._id_encoded_cols[name][pk] = encoder

            for id_col in self._id_value_cols[name]:
                if pd.api.types.is_numeric_dtype(data[name][id_col]):
                    continue

                key_values = self._get_values(data, name, id_col)
                encoder = LabelEncoder()
                encoder.fit(key_values)
                self._id_encoded_cols[name][id_col] = encoder

        for name in self._table_names:
            for parent, fk, _ppk in self._parents[name]:
                if parent in self._id_encoded_cols and _ppk in self._id_encoded_cols[parent]:
                    self._id_encoded_cols[name][fk] = self._id_encoded_cols[parent][_ppk]

        # preprocessing
        self._tables = {}
        for name in self._table_names:
            output_cols = list(data[name].columns)
            df = data[name].copy().reset_index(drop=True)
            # tables without a primary key get an internal row-id
            if name in self._synthetic_pk:
                df[self._primary_key[name]] = np.arange(len(df)) + 1

            # label-encode id columns
            for col, encoder in self._id_encoded_cols[name].items():
                notna = df[col].notna()
                df.loc[notna, col] = encoder.transform(df.loc[notna, col].astype(str))
                df[col] = pd.to_numeric(df[col])

            # preprocess_utils: encode datetimes as days since the earliest date
            date_info = {}
            for col, datetime_format in self._datetime_cols[name].items():
                if datetime_format is None:
                    datetime_format = guess_array_datetime_format(df[col])
                    df[col] = pd.to_datetime(df[col], format='mixed').dt.strftime(datetime_format)

                days_since, earliest = calculate_days_since_earliest_date(df[col], datetime_format)
                df[col] = np.asarray(days_since, dtype=float)
                date_info[col] = (earliest, datetime_format)

            # preprocess_utils: label-encode the discrete columns, derive the domain
            df, label_encoders = table_label_encode(df, self._discrete_cols[name])
            domain = get_domain(
                df, self._id_cols(name) + self._id_value_cols[name], self._discrete_cols[name]
            )

            # impute missing values
            continuous_cols = [col for col, spec in domain.items() if spec['type'] == 'continuous']
            if continuous_cols:
                imputer = _DataTransformer(
                    {col: {'sdtype': 'numerical'} for col in continuous_cols},
                    normalization=None,
                    seed=self.seed,
                )
                imputer.fit(df[continuous_cols])
                x_num, _ = imputer.transform(df[continuous_cols])
                df[continuous_cols] = x_num.astype(float)

            self._tables[name] = {
                'df': df,
                'domain': domain,
                'output_cols': output_cols,
                'original_len': len(data[name]),
                'pk': self._primary_key[name],
                'parents': list(self._parents[name]),
                'label_encoders': label_encoders,
                'date_info': date_info,
                'cluster_cols': [],
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

            cluster_col = f'{parent}_{child}_cluster'
            self._tables[parent]['cluster_cols'].append(cluster_col)
            self._tables[child]['cluster_cols'].append(cluster_col)
            for col in self._tables[parent]['cluster_cols']:
                parent_df[col] = parent_df[col].astype(int)
            for col in self._tables[child]['cluster_cols']:
                child_df[col] = child_df[col].astype(int)

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
            if column in domain:
                sdtype = 'categorical' if domain[column]['type'] == 'discrete' else 'numerical'
                columns_meta[column] = {'sdtype': sdtype}
            elif column in self._id_value_cols[child]:
                columns_meta[column] = {'sdtype': 'id'}

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
        model._skip_validation = True
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
                    self._id_cols(child) + self._id_value_cols[child],
                    n_clusters=self.num_matching_clusters,
                    unique_matching=self.unique_matching,
                    batch_size=self.matching_batch_size,
                    no_matching=self.no_matching,
                )
            else:
                final_tables[child] = synthetic_tables[(parent, child)]['df']

        return {child: self._postprocess(child, table) for child, table in final_tables.items()}

    def _postprocess(self, table, df):
        """Invert fit-time preprocessing: decode discretes, rebuild dates, decode ids."""
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

        # decode the primary/foreign key columns back to the original id values
        for col, encoder in self._id_encoded_cols[table].items():
            if col in df.columns:
                df[col] = decode_id_values(df[col].to_numpy(), encoder, col)

        return df[state['output_cols']].reset_index(drop=True)


class ClavaDDPMSynthesizer(MultiTableBaselineSynthesizer):
    """Custom wrapper for the ClavaDDPM synthesizer to make it work with SDGym."""

    LOGGER = logging.getLogger(__name__)
    _MODEL_KWARGS = None

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
