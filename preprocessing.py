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
            "f200", "f201", "f202", "f203", "f204", "f205", "f206",
            "f207", "f208", "f209", "f210", "f211", "f212", "f213",
            "f214", "f215", "f216", "f217", "f218", "f219", "f220", "f221", "f222",
            "f223", "f224", "f225", "f226", "f227", "f228", "f229", "f230", "f231",
            "f232", "f233", "f234", "f235", "f236", "f237", "f238", "f240", "f242", "f243", "f244", "f245", "f246", "f247", "f248"
        ]
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        self.tags = {"~"}
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
                    if c_word[-1] == '0' and any(ch.isdigit() for ch in c_word):
                        self.feature_rep_dict["f207"][(True, c_tag)] = self.feature_rep_dict["f207"].get((True, c_tag), 0) + 1

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
                    if sentence[i - 1][0].lower() in {"many", "few", "these", "those", "several", "some", "various", "numerous"}:
                        self.feature_rep_dict["f206"][(True, sentence[i][1])] = self.feature_rep_dict["f206"].get((True, sentence[i][1]), 0) + 1

                    if sentence[i + 1][0][0].isupper():
                        self.feature_rep_dict["f208"][(True, sentence[i][1])] = self.feature_rep_dict["f208"].get((True, sentence[i][1]), 0) + 1
                    if sentence[i + 1][0].isupper():
                        self.feature_rep_dict["f209"][(True, sentence[i][1])] = self.feature_rep_dict["f209"].get((True, sentence[i][1]), 0) + 1
                    if any(ch.isdigit() for ch in sentence[i + 1][0]):
                        self.feature_rep_dict["f210"][(True, sentence[i][1])] = self.feature_rep_dict["f210"].get((True, sentence[i][1]), 0) + 1
                    if '-' in sentence[i + 1][0]:
                        self.feature_rep_dict["f211"][(True, sentence[i][1])] = self.feature_rep_dict["f211"].get((True, sentence[i][1]), 0) + 1
                    if any(ch in string.punctuation for ch in sentence[i + 1][0]):
                        self.feature_rep_dict["f212"][(True, sentence[i][1])] = self.feature_rep_dict["f212"].get((True, sentence[i][1]), 0) + 1
                    if all(ch in string.punctuation for ch in sentence[i + 1][0]):
                        self.feature_rep_dict["f213"][(True, sentence[i][1])] = self.feature_rep_dict["f213"].get((True, sentence[i][1]), 0) + 1

                    if sentence[i - 1][0][0].isupper():
                        self.feature_rep_dict["f214"][(True, sentence[i][1])] = self.feature_rep_dict["f214"].get((True, sentence[i][1]), 0) + 1
                    if sentence[i - 1][0].isupper():
                        self.feature_rep_dict["f215"][(True, sentence[i][1])] = self.feature_rep_dict["f215"].get((True, sentence[i][1]), 0) + 1
                    if any(ch.isdigit() for ch in sentence[i - 1][0]):
                        self.feature_rep_dict["f216"][(True, sentence[i][1])] = self.feature_rep_dict["f216"].get((True, sentence[i][1]), 0) + 1
                    if '-' in sentence[i - 1][0]:
                        self.feature_rep_dict["f217"][(True, sentence[i][1])] = self.feature_rep_dict["f217"].get((True, sentence[i][1]), 0) + 1
                    if any(ch in string.punctuation for ch in sentence[i - 1][0]):
                        self.feature_rep_dict["f218"][(True, sentence[i][1])] = self.feature_rep_dict["f218"].get((True, sentence[i][1]), 0) + 1
                    if all(ch in string.punctuation for ch in sentence[i - 1][0]):
                        self.feature_rep_dict["f219"][(True, sentence[i][1])] = self.feature_rep_dict["f219"].get((True, sentence[i][1]), 0) + 1

                    if sentence[i + 1][0].endswith("ly"):
                        self.feature_rep_dict["f220"][(True, sentence[i][1])] = self.feature_rep_dict["f220"].get((True, sentence[i][1]), 0) + 1
                    if sentence[i + 1][0].endswith("ed") or sentence[i + 1][0].endswith("ing"):
                        self.feature_rep_dict["f221"][(True, sentence[i][1])] = self.feature_rep_dict["f221"].get((True, sentence[i][1]), 0) + 1
                    if sentence[i - 1][0].endswith("ed") or sentence[i - 1][0].endswith("ing"):
                        self.feature_rep_dict["f222"][(True, sentence[i][1])] = self.feature_rep_dict["f222"].get((True, sentence[i][1]), 0) + 1

                    if sentence[i - 1][1] == sentence[i - 2][1]:
                        self.feature_rep_dict["f223"][(True, sentence[i][1])] = self.feature_rep_dict["f223"].get((True, sentence[i][1]), 0) + 1

                    if sentence[i][0].endswith("s") and sentence[i - 1][1] == "JJ":
                        self.feature_rep_dict["f224"][(True, sentence[i][1])] = self.feature_rep_dict["f224"].get((True, sentence[i][1]), 0) + 1
                    if sentence[i][0].endswith("s") and sentence[i - 1][1] == "IN":
                        self.feature_rep_dict["f225"][(True, sentence[i][1])] = self.feature_rep_dict["f225"].get((True, sentence[i][1]), 0) + 1
                    if sentence[i - 2][1] == '*' and sentence[i - 1][1] == '*' and sentence[i][0][0].isupper():
                        self.feature_rep_dict["f226"][(True, sentence[i][1])] = self.feature_rep_dict["f226"].get((True, sentence[i][1]), 0) + 1

                    if any(t.startswith("NN") for t in [sentence[i - 1][1], sentence[i - 2][1]]):
                        self.feature_rep_dict["f227"][(True, sentence[i][1])] = self.feature_rep_dict["f227"].get((True, sentence[i][1]), 0) + 1
                    if any(t.startswith("VB") for t in [sentence[i - 1][1], sentence[i - 2][1]]):
                        self.feature_rep_dict["f228"][(True, sentence[i][1])] = self.feature_rep_dict["f228"].get((True, sentence[i][1]), 0) + 1
                    if (sentence[i - 1][1] == "CD" and sentence[i - 1][0] not in {"1", "one"}) or \
                    (sentence[i - 2][1] == "CD" and sentence[i - 2][0] not in {"1", "one"}) or \
                    any(t in ["NNS", "NNPS"] for t in [sentence[i - 1][1], sentence[i - 2][1]]):
                        self.feature_rep_dict["f229"][(True, sentence[i][1])] = self.feature_rep_dict["f229"].get((True, sentence[i][1]), 0) + 1

                    if sentence[i + 1][0].lower() in {"a", "an", "the"}:
                        key = (sentence[i + 1][0].lower(), sentence[i][1])
                        self.feature_rep_dict["f230"][key] = self.feature_rep_dict["f230"].get(key, 0) + 1

                    if sentence[i + 1][0] == ".":
                        self.feature_rep_dict["f231"][(True, sentence[i][1])] = self.feature_rep_dict["f231"].get((True, sentence[i][1]), 0) + 1
                        key = sentence[i - 1][1]
                        self.feature_rep_dict["f232"][key] = self.feature_rep_dict["f232"].get(key, 0) + 1

                    if sentence[i - 1][1] == sentence[i - 2][1]:
                        key = sentence[i - 1][1]
                        self.feature_rep_dict["f233"][key] = self.feature_rep_dict["f233"].get(key, 0) + 1

                    if sentence[i][1] in [sentence[i - 1][1], sentence[i - 2][1]]:
                        key = sentence[i][1]
                        self.feature_rep_dict["f234"][key] = self.feature_rep_dict["f234"].get(key, 0) + 1

                    if sentence[i][0].endswith("ly"):
                        self.feature_rep_dict["f235"][(True, sentence[i][1])] = self.feature_rep_dict["f235"].get((True, sentence[i][1]), 0) + 1

                    # f236: endswith 'ly' and tag is adverb
                    if sentence[i][0].endswith("ly") and sentence[i][1] in {"RB", "RBR", "RBS"}:
                        self.feature_rep_dict["f236"][(True, sentence[i][1])] = self.feature_rep_dict["f236"].get((True, sentence[i][1]), 0) + 1

                    # f237: endswith 'ly' and tag is adjective
                    if sentence[i][0].endswith("ly") and sentence[i][1] in {"JJ", "JJR", "JJS"}:
                        self.feature_rep_dict["f237"][(True, sentence[i][1])] = self.feature_rep_dict["f237"].get((True, sentence[i][1]), 0) + 1

                    # f238: two verbs next to each other
                    if sentence[i][1].startswith("VB") and sentence[i - 1][1].startswith("VB"):
                        self.feature_rep_dict["f238"][(True, sentence[i][1])] = self.feature_rep_dict["f238"].get((True, sentence[i][1]), 0) + 1

       
                    # f240: 'is/was/has/have/were' preceded by a verb
                    if sentence[i][0].lower() in {"is", "was", "has", "have", "were"} and sentence[i - 1][1].startswith("VB"):
                        self.feature_rep_dict["f240"][(True, sentence[i][0].lower())] = self.feature_rep_dict["f240"].get((True, sentence[i][0].lower()), 0) + 1

  
                    # f242: adverb follows a verb
                    if sentence[i - 1][1].startswith("VB") and sentence[i][1].startswith("RB"):
                        self.feature_rep_dict["f242"][(True, sentence[i][1])] = self.feature_rep_dict["f242"].get((True, sentence[i][1]), 0) + 1

                    # f243: verb follows an adverb
                    if sentence[i - 1][1].startswith("RB") and sentence[i][1].startswith("VB"):
                        self.feature_rep_dict["f243"][(True, sentence[i][1])] = self.feature_rep_dict["f243"].get((True, sentence[i][1]), 0) + 1

                    if sentence[i][1] == "NNPS" and sentence[i][0].endswith("s") and any(ch.isupper() for ch in sentence[i][0]):
                        self.feature_rep_dict["f244"][(True, sentence[i][1])] = self.feature_rep_dict["f244"].get((True, sentence[i][1]), 0) + 1

                    if sentence[i][0].endswith("ing") and sentence[i][1] == "VBG":
                        self.feature_rep_dict["f245"][(True, sentence[i][1])] = self.feature_rep_dict["f245"].get((True, sentence[i][1]), 0) + 1

                    if sentence[i][0].endswith("ed") and sentence[i][1] == "VBD":
                        self.feature_rep_dict["f246"][(True, sentence[i][1])] = self.feature_rep_dict["f246"].get((True, sentence[i][1]), 0) + 1

                        
                    if sentence[i][1] == "NNP" and sentence[i][0][0].isupper() and not sentence[i][0].endswith("s") and sentence[i - 1][0] != "*":
                        self.feature_rep_dict["f247"][(True, sentence[i][1])] = self.feature_rep_dict["f247"].get((True, sentence[i][1]), 0) + 1

                    adj_suffixes = ("able", "ible", "ous", "ful", "nal")
                    if sentence[i][1] in {"JJ", "JJR", "JJS"} and sentence[i][0].endswith(adj_suffixes):
                        self.feature_rep_dict["f248"][(True, sentence[i][1])] = self.feature_rep_dict["f248"].get((True, sentence[i][1]), 0) + 1

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
    import string

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

    # Features f200 to f207 (characteristic features)
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

    if p_word.lower() in {"many", "few", "these", "those", "several", "some", "various", "numerous"} and (True, c_tag) in dict_of_dicts["f206"]:
        features.append(dict_of_dicts["f206"][(True, c_tag)])

    if c_word[-1] == '0' and any(ch.isdigit() for ch in c_word) and (True, c_tag) in dict_of_dicts["f207"]:
        features.append(dict_of_dicts["f207"][(True, c_tag)])

    if n_word[0].isupper() and (True, c_tag) in dict_of_dicts["f208"]:
        features.append(dict_of_dicts["f208"][(True, c_tag)])
    if n_word.isupper() and (True, c_tag) in dict_of_dicts["f209"]:
        features.append(dict_of_dicts["f209"][(True, c_tag)])
    if any(ch.isdigit() for ch in n_word) and (True, c_tag) in dict_of_dicts["f210"]:
        features.append(dict_of_dicts["f210"][(True, c_tag)])
    if '-' in n_word and (True, c_tag) in dict_of_dicts["f211"]:
        features.append(dict_of_dicts["f211"][(True, c_tag)])
    if any(ch in string.punctuation for ch in n_word) and (True, c_tag) in dict_of_dicts["f212"]:
        features.append(dict_of_dicts["f212"][(True, c_tag)])
    if all(ch in string.punctuation for ch in n_word) and (True, c_tag) in dict_of_dicts["f213"]:
        features.append(dict_of_dicts["f213"][(True, c_tag)])

    if p_word[0].isupper() and (True, c_tag) in dict_of_dicts["f214"]:
        features.append(dict_of_dicts["f214"][(True, c_tag)])
    if p_word.isupper() and (True, c_tag) in dict_of_dicts["f215"]:
        features.append(dict_of_dicts["f215"][(True, c_tag)])
    if any(ch.isdigit() for ch in p_word) and (True, c_tag) in dict_of_dicts["f216"]:
        features.append(dict_of_dicts["f216"][(True, c_tag)])
    if '-' in p_word and (True, c_tag) in dict_of_dicts["f217"]:
        features.append(dict_of_dicts["f217"][(True, c_tag)])
    if any(ch in string.punctuation for ch in p_word) and (True, c_tag) in dict_of_dicts["f218"]:
        features.append(dict_of_dicts["f218"][(True, c_tag)])
    if all(ch in string.punctuation for ch in p_word) and (True, c_tag) in dict_of_dicts["f219"]:
        features.append(dict_of_dicts["f219"][(True, c_tag)])

    if n_word.endswith("ly") and (True, c_tag) in dict_of_dicts["f220"]:
        features.append(dict_of_dicts["f220"][(True, c_tag)])
    if (n_word.endswith("ed") or n_word.endswith("ing")) and (True, c_tag) in dict_of_dicts["f221"]:
        features.append(dict_of_dicts["f221"][(True, c_tag)])
    if (p_word.endswith("ed") or p_word.endswith("ing")) and (True, c_tag) in dict_of_dicts["f222"]:
        features.append(dict_of_dicts["f222"][(True, c_tag)])

    if p_tag == pp_tag and (True, c_tag) in dict_of_dicts["f223"]:
        features.append(dict_of_dicts["f223"][(True, c_tag)])
    if c_word.endswith("s") and p_tag == "JJ" and (True, c_tag) in dict_of_dicts["f224"]:
        features.append(dict_of_dicts["f224"][(True, c_tag)])
    if c_word.endswith("s") and p_tag == "IN" and (True, c_tag) in dict_of_dicts["f225"]:
        features.append(dict_of_dicts["f225"][(True, c_tag)])

    if pp_tag == '*' and p_tag == '*' and c_word[0].isupper():
        if 'f226' in dict_of_dicts and (True, c_tag) in dict_of_dicts['f226']:
            features.append(dict_of_dicts['f226'][(True, c_tag)])

    if any(t.startswith("NN") for t in [p_tag, pp_tag]):
        if 'f227' in dict_of_dicts and (True, c_tag) in dict_of_dicts['f227']:
            features.append(dict_of_dicts['f227'][(True, c_tag)])

    if any(t.startswith("VB") for t in [p_tag, pp_tag]):
        if 'f228' in dict_of_dicts and (True, c_tag) in dict_of_dicts['f228']:
            features.append(dict_of_dicts['f228'][(True, c_tag)])

    if (p_tag == "CD" and p_word not in {"1", "one"}) or (pp_tag == "CD" and pp_word not in {"1", "one"}) or \
        any(t in ["NNS", "NNPS"] for t in [p_tag, pp_tag]):
        if 'f229' in dict_of_dicts and (True, c_tag) in dict_of_dicts['f229']:
            features.append(dict_of_dicts['f229'][(True, c_tag)])

    if n_word.lower() in {"a", "an", "the"}:
        if 'f230' in dict_of_dicts and (n_word.lower(), c_tag) in dict_of_dicts['f230']:
            features.append(dict_of_dicts['f230'][(n_word.lower(), c_tag)])

    if n_word == ".":
        if 'f231' in dict_of_dicts and (True, c_tag) in dict_of_dicts['f231']:
            features.append(dict_of_dicts['f231'][(True, c_tag)])

    if (p_tag == "CD" and p_word not in {"1", "one"}) or (pp_tag == "CD" and pp_word not in {"1", "one"}) or \
    any(t in ["NNS", "NNPS"] for t in [p_tag, pp_tag]):
        if 'f229' in dict_of_dicts and (True, c_tag) in dict_of_dicts['f229']:
            features.append(dict_of_dicts['f229'][(True, c_tag)])

    if n_word.lower() in {"a", "an", "the"}:
        if 'f230' in dict_of_dicts and (n_word.lower(), c_tag) in dict_of_dicts['f230']:
            features.append(dict_of_dicts['f230'][(n_word.lower(), c_tag)])

    if n_word == ".":
        if 'f231' in dict_of_dicts and (True, c_tag) in dict_of_dicts['f231']:
            features.append(dict_of_dicts['f231'][(True, c_tag)])

    if n_word == ".":
        if 'f232' in dict_of_dicts and p_tag in dict_of_dicts['f232']:
            features.append(dict_of_dicts['f232'][p_tag])

    if p_tag == pp_tag:
        if 'f233' in dict_of_dicts and p_tag in dict_of_dicts['f233']:
            features.append(dict_of_dicts['f233'][p_tag])

    if c_tag in [p_tag, pp_tag]:
        if 'f234' in dict_of_dicts and c_tag in dict_of_dicts['f234']:
            features.append(dict_of_dicts['f234'][c_tag])
    
        # f235: current word ends in "ly"
    if c_word.endswith("ly") and (True, c_tag) in dict_of_dicts.get("f235", {}):
        features.append(dict_of_dicts["f235"][(True, c_tag)])

    # f236: ends with "ly" and tag is adverb
    if c_word.endswith("ly") and (True, c_tag) in dict_of_dicts.get("f236", {}):
        features.append(dict_of_dicts["f236"][(True, c_tag)])

    # f237: ends with "ly" and tag is adjective
    if c_word.endswith("ly") and (True, c_tag) in dict_of_dicts.get("f237", {}):
        features.append(dict_of_dicts["f237"][(True, c_tag)])

    # f238: two verbs next to each other
    if p_tag.startswith("VB") and c_tag.startswith("VB") and (True, c_tag) in dict_of_dicts.get("f238", {}):
        features.append(dict_of_dicts["f238"][(True, c_tag)])


    # f240: current word is 'is', 'was', etc., and previous is a verb
    if c_word.lower() in {"is", "was", "has", "have", "were"} and p_tag.startswith("VB"):
        if (True, c_word.lower()) in dict_of_dicts.get("f240", {}):
            features.append(dict_of_dicts["f240"][(True, c_word.lower())])

    # f242: adverb follows a verb
    if p_tag.startswith("VB") and c_tag.startswith("RB") and (True, c_tag) in dict_of_dicts.get("f242", {}):
        features.append(dict_of_dicts["f242"][(True, c_tag)])

    # f243: verb follows an adverb
    if p_tag.startswith("RB") and c_tag.startswith("VB") and (True, c_tag) in dict_of_dicts.get("f243", {}):
        features.append(dict_of_dicts["f243"][(True, c_tag)])

    if c_tag == "NNPS" and c_word.endswith("s") and any(ch.isupper() for ch in c_word):
        if (True, c_tag) in dict_of_dicts.get("f244", {}):
            features.append(dict_of_dicts["f244"][(True, c_tag)])


    if c_word.endswith("ing") and c_tag == "VBG":
        if (True, c_tag) in dict_of_dicts.get("f245", {}):
            features.append(dict_of_dicts["f245"][(True, c_tag)])

    if c_word.endswith("ed") and c_tag == "VBD":
        if (True, c_tag) in dict_of_dicts.get("f246", {}):
            features.append(dict_of_dicts["f246"][(True, c_tag)])

    if c_tag == "NNP" and c_word[0].isupper() and not c_word.endswith("s") and p_word != "*":
        if (True, c_tag) in dict_of_dicts.get("f247", {}):
            features.append(dict_of_dicts["f247"][(True, c_tag)])

    adj_suffixes = ("able", "ible", "ous", "ful", "nal")
    if c_tag in {"JJ", "JJR", "JJS"} and c_word.endswith(adj_suffixes):
        if (True, c_tag) in dict_of_dicts.get("f248", {}):
            features.append(dict_of_dicts["f248"][(True, c_tag)])


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
