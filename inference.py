from preprocessing import represent_input_with_features
from preprocessing import read_test
from tqdm import tqdm

import numpy as np
from collections import defaultdict

def memm_viterbi(sentence, pre_trained_weights, feature2id):
    _beamwidth = 20
    _tags = feature2id.feature_statistics.tags
    n = len(sentence)
    _beam = [(['*', '*'], 0)]  # Initial beam: dummy tags and log-prob 0

    for t in range(2, n - 1):  # Start at third token ("real" sentence start)
        _beamcandidates = []
        for _knowntags, _score in _beam:
            _tagscores = []
            for _tag in _tags:
                _history = (
                    sentence[t],
                    _tag,
                    sentence[t - 1],
                    _knowntags[-1],
                    sentence[t - 2],
                    _knowntags[-2],
                    sentence[t + 1],
                )
                feature_indices = represent_input_with_features(_history, feature2id.feature_to_idx)
                _scoreval = sum(pre_trained_weights[idx] for idx in feature_indices)
                _tagscores.append((_tag, _scoreval))

            _maxscore = max(score for _, score in _tagscores)
            exp_scores = [(tag, np.exp(score - _maxscore)) for tag, score in _tagscores]
            _denominator = sum(score for _, score in exp_scores)

            for tag, unnormalized_score in _tagscores:
                prob = np.exp(unnormalized_score - _maxscore) / _denominator
                logprob = np.log(prob)
                new_known_tags = _knowntags + [tag]
                new_score = _score + logprob
                _beamcandidates.append((new_known_tags, new_score))

        _beamcandidates.sort(key=lambda x: x[1], reverse=True)
        _beam = _beamcandidates[:_beamwidth]

    best_sequence = max(_beam, key=lambda x: x[1])[0]
    return best_sequence[1:]  # Remove the two initial "*" dummy tags




def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()