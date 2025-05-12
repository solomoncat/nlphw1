from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple

WORD = 0
TAG = 1
from collections import OrderedDict, defaultdict
import string
from typing import List, Dict, Tuple

WORD = 0
TAG = 1

class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0
        feature_dict_list = [
            "f100", "f101", "f102", "f103", "f104", "f105",
            "f106", "f107", "f108",
            "f200", "f201", "f202", "f203", "f204", "f205"
        ]
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        self.tags = set()
        self.tags.add("~")
        self.tags_counts = defaultdict(int)
        self.words_count = defaultdict(int)
        self.histories = []

    def get_word_tag_pair_count(self, file_path) -> None:
        with open(file_path) as file:
            for line in file:
                line = line.strip()
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    c_word, c_tag = split_words[word_idx].split('_')
                    self.tags.add(c_tag)
                    self.tags_counts[c_tag] += 1
                    self.words_count[c_word] += 1
                    self.feature_rep_dict["f100"][(c_word, c_tag)] = self.feature_rep_dict["f100"].get((c_word, c_tag), 0) + 1

                    if c_word[0].isupper():
                        self.feature_rep_dict["f200"][(True, c_tag)] = self.feature_rep_dict["f200"].get((True, c_tag), 0) + 1
                    if c_word.isupper():
                        self.feature_rep_dict["f201"][(True, c_tag)] = self.feature_rep_dict["f201"].get((True, c_tag), 0) + 1
                    if c_word.isdigit():
                        self.feature_rep_dict["f202"][(True, c_tag)] = self.feature_rep_dict["f202"].get((True, c_tag), 0) + 1
                    if '-' in c_word:
                        self.feature_rep_dict["f203"][(True, c_tag)] = self.feature_rep_dict["f203"].get((True, c_tag), 0) + 1
                    if self.words_count[c_word] < 3:
                        self.feature_rep_dict["f204"][(True, c_tag)] = self.feature_rep_dict["f204"].get((True, c_tag), 0) + 1
                    if all(ch in string.punctuation for ch in c_word):
                        self.feature_rep_dict["f205"][(True, c_tag)] = self.feature_rep_dict["f205"].get((True, c_tag), 0) + 1

                # Add boundary markers
                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1],
                        sentence[i - 2][0], sentence[i - 2][1], sentence[i + 1][0]
                    )
                    self.histories.append(history)

                    trigram = (sentence[i - 2][1], sentence[i - 1][1], sentence[i][1])
                    self.feature_rep_dict["f103"][trigram] = self.feature_rep_dict["f103"].get(trigram, 0) + 1

                    digram = (sentence[i - 1][1], sentence[i][1])
                    self.feature_rep_dict["f104"][digram] = self.feature_rep_dict["f104"].get(digram, 0) + 1

                    monogram = sentence[i][1]
                    self.feature_rep_dict["f105"][monogram] = self.feature_rep_dict["f105"].get(monogram, 0) + 1

                    if sentence[i - 1][0] == "the":
                        key = (sentence[i - 1][0], sentence[i][1])
                        self.feature_rep_dict["f106"][key] = self.feature_rep_dict["f106"].get(key, 0) + 1

                    key = (sentence[i + 1][0], sentence[i][1])
                    if sentence[i + 1][0] == "the":
                        self.feature_rep_dict["f107"][key] = self.feature_rep_dict["f107"].get(key, 0) + 1
                    if sentence[i + 1][0].isupper():
                        self.feature_rep_dict["f108"][key] = self.feature_rep_dict["f108"].get(key, 0) + 1

    def get_prefix_count(self, file_path) -> None:
        with open(file_path) as file:
            for line in file:
                line = line.strip()
                for word_tag in line.split(' '):
                    word = word_tag.split('_')[0]
                    for i in range(1, 5):
                        prefix = word[:i]
                        self.feature_rep_dict["f102"][prefix] = self.feature_rep_dict["f102"].get(prefix, 0) + 1

    def get_suffix_count(self, file_path) -> None:
        with open(file_path) as file:
            for line in file:
                line = line.strip()
                for word_tag in line.split(' '):
                    word = word_tag.split('_')[0]
                    for i in range(1, 5):
                        suffix = word[-i:]
                        self.feature_rep_dict["f101"][suffix] = self.feature_rep_dict["f101"].get(suffix, 0) + 1


class Feature2id:
    def __init__(self):
        self.feature_rep_dict = {}
        self.total = 0

def preprocess_train(file_path: str, threshold_dict: Dict[str, int]) -> Tuple[FeatureStatistics, Feature2id]:
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(file_path)
    statistics.get_prefix_count(file_path)
    statistics.get_suffix_count(file_path)
    for fname, rep in statistics.feature_rep_dict.items():
        print(f"\nFeature class: {fname}")
    for feat, count in rep.items():
        print(f"  {feat}: {count}")

    feature2id = Feature2id()

    for fname, rep in statistics.feature_rep_dict.items():
        threshold = threshold_dict.get(fname, 20)  # Default to 20 if not specified
        filtered_rep = {k: v for k, v in rep.items() if v >= threshold}
        feature2id.feature_rep_dict[fname] = filtered_rep
        feature2id.total += len(filtered_rep)

    return statistics, feature2id

class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold_f1: int = 20, threshold_f2: int = 50):
        self.feature_statistics = feature_statistics
        self.n_total_features = 0
        self.threshold_f1 = threshold_f1
        self.threshold_f2 = threshold_f2

        self.feature_to_idx = {k: OrderedDict() for k in feature_statistics.feature_rep_dict}
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        for feat_class, feat_dict in self.feature_statistics.feature_rep_dict.items():
            threshold = self.threshold_f1 if feat_class.startswith("f1") else self.threshold_f2
            for feat, count in feat_dict.items():
                if count >= threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        from preprocessing import represent_input_with_features  # avoids circular import
        big_r = 0
        big_rows, big_cols = [], []
        small_rows, small_cols = [], []

        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for y_tag in self.feature_statistics.tags:
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1

        self.big_matrix = sparse.csr_matrix(
            (np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
            shape=(len(self.feature_statistics.tags) * len(self.feature_statistics.histories), self.n_total_features),
            dtype=bool
        )
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(self.feature_statistics.histories), self.n_total_features),
            dtype=bool
        )

def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple, int]]) -> List[int]:
    """
    Extract feature vector given a history tuple and feature dictionaries.

    @param history: tuple{c_word, c_tag, p_word, p_tag, n_word, pp_tag, pp_word}
    @param dict_of_dicts: a dictionary of each feature and the index it was given
    @return: a list with all relevant features for the given history
    """
    c_word, c_tag, p_word, p_tag, n_word, pp_tag, pp_word = history

    features = []

    # Features f100 to f108 (contextual features)
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    for i in range(1, 5):
        suffix = c_word[-i:]
        if suffix in dict_of_dicts['f101']:
            features.append(dict_of_dicts['f101'][suffix])

    for i in range(1, 5):
        prefix = c_word[:i]
        if prefix in dict_of_dicts['f102']:
            features.append(dict_of_dicts['f102'][prefix])

    tag_trigram = (pp_tag, p_tag, c_tag)
    if tag_trigram in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][tag_trigram])

    tag_digram = (p_tag, c_tag)
    if tag_digram in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][tag_digram])

    tag_monogram = (c_tag,)
    if tag_monogram in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][tag_monogram])

    wordtag_prev = (p_word, c_tag)
    if wordtag_prev in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][wordtag_prev])

    tagword_next = (n_word, c_tag)
    if tagword_next in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][tagword_next])

    if tagword_next in dict_of_dicts["f108"]:
        features.append(dict_of_dicts["f108"][tagword_next])

    # Features f200 to f205 (characteristic features)
    if c_word[0].isupper() and (True, c_tag) in dict_of_dicts["f200"]:
        features.append(dict_of_dicts["f200"][(True, c_tag)])

    if c_word.isupper() and (True, c_tag) in dict_of_dicts["f201"]:
        features.append(dict_of_dicts["f201"][(True, c_tag)])

    if c_word.isdigit() and (True, c_tag) in dict_of_dicts["f202"]:
        features.append(dict_of_dicts["f202"][(True, c_tag)])

    if '-' in c_word and (True, c_tag) in dict_of_dicts["f203"]:
        features.append(dict_of_dicts["f203"][(True, c_tag)])

    if (True, c_tag) in dict_of_dicts["f204"]:
        features.append(dict_of_dicts["f204"][(True, c_tag)])

    if all(ch in string.punctuation for ch in c_word) and (True, c_tag) in dict_of_dicts["f205"]:
        features.append(dict_of_dicts["f205"][(True, c_tag)])

    return features


def preprocess_train(train_path: str, threshold_f1: int, threshold_f2: int):
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)
    statistics.get_prefix_count(train_path)
    statistics.get_suffix_count(train_path)

    feature2id = Feature2id(statistics, threshold_f1, threshold_f2)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()

    return statistics, feature2id

def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            sentence = (["*", "*"], ["*", "*"])
            for word_tag in line.split(' '):
                if tagged:
                    word, tag = word_tag.split('_')
                else:
                    word, tag = word_tag, ""
                sentence[WORD].append(word)
                sentence[TAG].append(tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences