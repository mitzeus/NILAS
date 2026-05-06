from collections import Counter
import numpy as np

STRUCTURAL_FEATURES = {
    "sent_len_mean",
    "sent_len_std",
    "tree_depth_mean",
    "tree_depth_std",
    "avg_token_length",
}

RATIO_FEATURES = {
    "discourse_ratio",
    "type_token_ratio",
    "stopword_ratio",
    "punct_ratio",
}

LEXICAL_RATIO_FEATURES = {
    "type_token_ratio",
    "stopword_ratio",
    "punct_ratio",
}


def get_pos_distribution(doc):
    counts = Counter([token.pos_ for token in doc])
    total = sum(counts.values())
    return {f"pos_{k}": v / total for k, v in counts.items()}


def get_dep_distribution(doc):
    counts = Counter([token.dep_ for token in doc])
    total = sum(counts.values())
    return {f"dep_{k}": v / total for k, v in counts.items()}


def get_discourse_features(doc):
    # Simple heuristic discourse markers
    discourse_markers = {
        "however",
        "therefore",
        "moreover",
        "thus",
        "but",
        "and",
        "so",
        "because",
        "although",
    }
    tokens = [token.text.lower() for token in doc]
    count = sum(1 for t in tokens if t in discourse_markers)
    return {"discourse_ratio": count / max(len(tokens), 1)}


def get_sentence_length_stats(doc):
    lengths = [len(sent) for sent in doc.sents]
    if not lengths:
        return {"sent_len_mean": 0, "sent_len_std": 0}
    return {"sent_len_mean": np.mean(lengths), "sent_len_std": np.std(lengths)}


def get_tree_depth(token):
    depth = 0
    while token.head != token:
        token = token.head
        depth += 1
    return depth


def get_tree_depth_stats(doc):
    depths = [get_tree_depth(token) for token in doc]
    if not depths:
        return {"tree_depth_mean": 0, "tree_depth_std": 0}
    return {"tree_depth_mean": np.mean(depths), "tree_depth_std": np.std(depths)}


def get_lexical_features(doc):
    tokens = [t for t in doc if not t.is_space]
    words = [t.text.lower() for t in tokens if t.is_alpha]

    if not tokens:
        return {
            "type_token_ratio": 0,
            "avg_token_length": 0,
            "stopword_ratio": 0,
            "punct_ratio": 0,
        }

    return {
        "type_token_ratio": len(set(words)) / max(len(words), 1),
        "avg_token_length": np.mean([len(t.text) for t in tokens]),
        "stopword_ratio": sum(t.is_stop for t in tokens) / len(tokens),
        "punct_ratio": sum(t.is_punct for t in tokens) / len(tokens),
    }


def extract_features(text, feature_nlp):
    doc = feature_nlp(text)

    features = {}

    # Core features
    features.update(get_pos_distribution(doc))
    features.update(get_dep_distribution(doc))
    features.update(get_discourse_features(doc))
    features.update(get_sentence_length_stats(doc))
    features.update(get_tree_depth_stats(doc))

    # Extra recommended features
    features.update(get_lexical_features(doc))

    return features


def build_feature_matrix(texts, feature_nlp, keys=None):
    feature_dicts = [extract_features(t, feature_nlp) for t in texts]
    return build_feature_matrix_from_dicts(feature_dicts, keys)


def build_feature_matrix_from_dicts(feature_dicts, keys=None):
    all_keys = sorted(set().union(*feature_dicts)) if keys is None and feature_dicts else keys
    all_keys = [] if all_keys is None else all_keys

    matrix = np.zeros((len(feature_dicts), len(all_keys)))

    key_index = {key: i for i, key in enumerate(all_keys)}
    for row, fdict in enumerate(feature_dicts):
        for key, value in fdict.items():
            col = key_index.get(key)
            if col is not None:
                matrix[row, col] = value

    return matrix, all_keys


def _l1_normalize_columns(matrix, columns):
    if not columns:
        return

    block = matrix[:, columns]
    totals = block.sum(axis=1, keepdims=True)
    np.divide(block, totals, out=block, where=totals > 0)
    matrix[:, columns] = block


def get_feature_groups(keys):
    return {
        "pos": [i for i, key in enumerate(keys) if key.startswith("pos_")],
        "dep": [i for i, key in enumerate(keys) if key.startswith("dep_")],
        "discourse": [i for i, key in enumerate(keys) if key == "discourse_ratio"],
        "lexical": [i for i, key in enumerate(keys) if key in LEXICAL_RATIO_FEATURES],
        "structural": [i for i, key in enumerate(keys) if key in STRUCTURAL_FEATURES],
    }


def scale_feature_matrix(matrix, keys):
    scaled = matrix.astype(float, copy=True)
    feature_groups = get_feature_groups(keys)

    structural_columns = feature_groups["structural"]
    scalar_columns = [
        i
        for i, key in enumerate(keys)
        if key in STRUCTURAL_FEATURES or key in RATIO_FEATURES
    ]

    _l1_normalize_columns(scaled, feature_groups["pos"])
    _l1_normalize_columns(scaled, feature_groups["dep"])

    if structural_columns:
        scaled[:, structural_columns] = np.log1p(scaled[:, structural_columns])

    if scalar_columns:
        means = scaled[:, scalar_columns].mean(axis=0)
        stds = scaled[:, scalar_columns].std(axis=0)
        stds[stds == 0.0] = 1.0
        scaled[:, scalar_columns] = (scaled[:, scalar_columns] - means) / stds

    for columns in feature_groups.values():
        if columns:
            scaled[:, columns] /= np.sqrt(len(columns))

    return scaled


def _squared_distances(X, Y):
    XX = np.einsum("ij,ij->i", X, X)[:, None]
    YY = np.einsum("ij,ij->i", Y, Y)[None, :]
    return np.maximum(XX + YY - 2.0 * X @ Y.T, 0.0)


def _median_gamma(X, Y):
    distances = _squared_distances(np.vstack((X, Y)), np.vstack((X, Y)))
    distances = distances[distances > 0.0]
    if distances.size == 0:
        return 1.0
    return 1.0 / (2.0 * np.median(distances))


def rbf_kernel(X, Y, gamma):
    distances = _squared_distances(X, Y)
    return np.exp(-gamma * distances)


def compute_mmd(X, Y, gamma=None):
    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError("MMD requires at least one sample in each distribution.")

    if gamma is None:
        gamma = _median_gamma(X, Y)

    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)

    m = X.shape[0]
    n = Y.shape[0]

    return np.sum(Kxx) / (m * m) + np.sum(Kyy) / (n * n) - 2 * np.sum(Kxy) / (m * n)


def compute_mmd_between_distributions(
    texts_A,
    texts_B,
    feature_nlp,
    gamma=None,
    return_components=False,
):
    features_A = [extract_features(text, feature_nlp) for text in texts_A]
    features_B = [extract_features(text, feature_nlp) for text in texts_B]
    all_keys = sorted(set().union(*features_A, *features_B))

    X, _ = build_feature_matrix_from_dicts(features_A, all_keys)
    Y, _ = build_feature_matrix_from_dicts(features_B, all_keys)
    combined = scale_feature_matrix(np.vstack((X, Y)), all_keys)
    X_scaled = combined[: len(X)]
    Y_scaled = combined[len(X) :]
    overall = compute_mmd(X_scaled, Y_scaled, gamma=gamma)

    if not return_components:
        return overall

    components = {"overall": overall}
    for name, columns in get_feature_groups(all_keys).items():
        if columns:
            components[name] = compute_mmd(
                X_scaled[:, columns],
                Y_scaled[:, columns],
                gamma=gamma,
            )

    return components
