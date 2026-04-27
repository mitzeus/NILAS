from sklearn.metrics.pairwise import rbf_kernel
from collections import Counter, defaultdict
import numpy as np

# TODO do strong typing and descriptions


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


def build_feature_matrix(texts, feature_nlp):
    feature_dicts = [extract_features(t, feature_nlp) for t in texts]

    # Collect all feature keys
    all_keys = sorted(set().union(*feature_dicts))

    matrix = np.zeros((len(texts), len(all_keys)))

    for i, fdict in enumerate(feature_dicts):
        for j, key in enumerate(all_keys):
            matrix[i, j] = fdict.get(key, 0.0)

    return matrix, all_keys


def rbf_kernel(X, Y, gamma=1.0):
    XX = np.sum(X**2, axis=1).reshape(-1, 1)
    YY = np.sum(Y**2, axis=1).reshape(1, -1)
    distances = XX + YY - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * distances)


def compute_mmd(X, Y, gamma=1.0):
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)

    m = X.shape[0]
    n = Y.shape[0]

    mmd = np.sum(Kxx) / (m * m) + np.sum(Kyy) / (n * n) - 2 * np.sum(Kxy) / (m * n)

    return mmd


def compute_mmd_between_distributions(texts_A, texts_B, feature_nlp, gamma=1.0):
    X, keys_X = build_feature_matrix(texts_A, feature_nlp)
    Y, keys_Y = build_feature_matrix(texts_B, feature_nlp)

    # Align feature spaces
    all_keys = sorted(set(keys_X).union(keys_Y))

    def align_matrix(matrix, old_keys, new_keys):
        key_index = {k: i for i, k in enumerate(old_keys)}
        aligned = np.zeros((matrix.shape[0], len(new_keys)))
        for j, k in enumerate(new_keys):
            if k in key_index:
                aligned[:, j] = matrix[:, key_index[k]]
        return aligned

    X_aligned = align_matrix(X, keys_X, all_keys)
    Y_aligned = align_matrix(Y, keys_Y, all_keys)

    return compute_mmd(X_aligned, Y_aligned, gamma=gamma)
