from collections import Counter
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# TODO do strong typing and descriptions


def ngrams(tokens, n):
    if n <= 0:
        return []
    return zip(*(tokens[i:] for i in range(n)))


def extract_features(
    texts: list[str],
    discourse_markers: list[str],
    n_gram_size: int,
    word_processor: object,
):  # extract per corpus
    pos_counts = Counter()
    dep_counts = Counter()
    discourse_counts = Counter()

    sentence_lengths = []
    tree_depths = []
    all_ngrams = Counter()

    for doc in word_processor.pipe(texts):
        for sent in doc.sents:
            tokens = [t.text.lower() for t in sent if not t.is_punct]

            # sentence length
            sentence_lengths.append(len(tokens))

            # POS + dependency
            for token in sent:
                pos_counts[token.pos_] += 1
                dep_counts[token.dep_] += 1

            # discourse markers
            for word in tokens:
                if word in discourse_markers:
                    discourse_counts[word] += 1

            # n-grams (bigrams)
            for bg in ngrams(tokens, n_gram_size):
                all_ngrams[bg] += 1

            # dependency tree depth
            def get_depth(token):
                if not list(token.children):
                    return 1
                return 1 + max(get_depth(child) for child in token.children)

            root = sent.root
            tree_depths.append(get_depth(root))

    return {
        "pos": pos_counts,
        "dep": dep_counts,
        "discourse": discourse_counts,
        "ngrams": all_ngrams,
        "sent_len": np.array(sentence_lengths),
        "tree_depth": np.array(tree_depths),
    }


# Convert to probability distribution
def normalize(counter):
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def jsd(p, q):  # Jensen-Shannon Divergence

    keys = list(set(p.keys()).union(set(q.keys())))
    if not keys:
        return 0.0

    p_vec = np.array([p.get(k, 0) for k in keys])
    q_vec = np.array([q.get(k, 0) for k in keys])

    # normalize again (safety)
    p_sum = p_vec.sum()
    q_sum = q_vec.sum()
    if p_sum == 0 and q_sum == 0:
        return 0.0
    if p_sum == 0 or q_sum == 0:
        return 1.0

    p_vec /= p_sum
    q_vec /= q_sum

    return jensenshannon(p_vec, q_vec)


def safe_wasserstein(values_A, values_B):
    if len(values_A) == 0 and len(values_B) == 0:
        return 0.0
    if len(values_A) == 0 or len(values_B) == 0:
        return np.nan
    return wasserstein_distance(values_A, values_B)


def calculate_JS_wasserstein(
    A: list[str],
    B: list[str],
    discourse_markers: list[str],
    n_gram_size: int,
    word_processor: object,
):
    features_A = extract_features(A, discourse_markers, n_gram_size, word_processor)
    features_B = extract_features(B, discourse_markers, n_gram_size, word_processor)

    # Normalize discrete
    pos_A = normalize(features_A["pos"])
    pos_B = normalize(features_B["pos"])

    dep_A = normalize(features_A["dep"])
    dep_B = normalize(features_B["dep"])

    disc_A = normalize(features_A["discourse"])
    disc_B = normalize(features_B["discourse"])

    ng_A = normalize(features_A["ngrams"])
    ng_B = normalize(features_B["ngrams"])

    # Compute divergences
    results = {
        "pos_jsd": jsd(pos_A, pos_B),
        "dep_jsd": jsd(dep_A, dep_B),
        "discourse_jsd": jsd(disc_A, disc_B),
        "ngram_jsd": jsd(ng_A, ng_B),
        "sent_len_wasserstein": safe_wasserstein(
            features_A["sent_len"], features_B["sent_len"]
        ),
        "tree_depth_wasserstein": safe_wasserstein(
            features_A["tree_depth"], features_B["tree_depth"]
        ),
    }

    # print(results)
    return results
