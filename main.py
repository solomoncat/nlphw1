import pickle
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import defaultdict

from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test

def compute_accuracy_from_files(predictions_path: str, gold_path: str, pred_start_line: int = 0):
    correct = total = 0
    sentence_accuracies = []
    all_true_tags = []
    all_pred_tags = []

    with open(predictions_path, 'r') as pred_file:
        pred_lines = pred_file.readlines()
    with open(gold_path, 'r') as gold_file:
        gold_lines = gold_file.readlines()

    # Slice predictions if needed
    pred_lines = pred_lines[pred_start_line:]

    for i, (pred_line, gold_line) in enumerate(zip(pred_lines, gold_lines)):
        pred_tokens = pred_line.strip().split()
        gold_tokens = gold_line.strip().split()

        pred_tags = [tok.split('_')[1] for tok in pred_tokens]
        gold_tags = [tok.split('_')[1] for tok in gold_tokens]

        assert len(pred_tags) == len(gold_tags), f"Mismatch at sentence {i}: {len(pred_tags)} vs {len(gold_tags)}"

        sent_correct = sum(p == g for p, g in zip(pred_tags, gold_tags))
        sent_total = len(gold_tags)
        correct += sent_correct
        total += sent_total

        all_true_tags.extend(gold_tags)
        all_pred_tags.extend(pred_tags)

        acc = sent_correct / sent_total if sent_total else 0.0
        sentence_accuracies.append(acc)

        if (i + 1) % 100 == 0:
            print(f"Sentence {i + 1} Accuracy: {acc:.4f}")

    overall = correct / total if total else 0.0
    return overall, sentence_accuracies, all_true_tags, all_pred_tags
def main():
    _time = time.time()
    threshold_f1 = 11
    threshold_f2 = 3
    lams = [1]
    train_path = "data/train1.wtag"
    test_path = "data/test1.wtag"
    predictions_path = "predictions.wtag"

    results = {}
    all_true_tags = []
    all_pred_tags = []

    for lam in lams:
        print(f"\n--- Running with lambda = {lam}, f1 threshold = {threshold_f1}, f2 threshold = {threshold_f2} ---")
        weights_path = f'weights_lam_{lam}_f1_{threshold_f1}_f2_{threshold_f2}.pkl'

        statistics, feature2id = preprocess_train(train_path, threshold_f1, threshold_f2)
        get_optimal_vector(statistics=statistics, feature2id=feature2id,
                           weights_path=weights_path, lam=lam)

        with open(weights_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]

        tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
        overall_accuracy, sentence_accuracies, true_tags, pred_tags = compute_accuracy_from_files(predictions_path, test_path)
        all_true_tags.extend(true_tags)
        all_pred_tags.extend(pred_tags)

        results[lam] = overall_accuracy

        for i, acc in enumerate(sentence_accuracies):
            print(f"Sentence {i + 1} Accuracy: {acc:.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")

    print("\n=== Summary of Accuracies by Lambda and Thresholds ===")
    for lam in lams:
        print(f"Lambda {lam}, f1 = {threshold_f1}, f2 = {threshold_f2}: Accuracy = {results[lam]:.4f}")

    # --- Per-tag error analysis ---
    print("\n=== Per-Tag Error Rates ===")
    from collections import defaultdict

    tag_total = defaultdict(int)
    tag_mistakes = defaultdict(int)

    for true_tag, pred_tag in zip(all_true_tags, all_pred_tags):
        tag_total[true_tag] += 1
        if true_tag != pred_tag:
            tag_mistakes[true_tag] += 1

    tag_error_stats = []
    for tag in sorted(tag_total):
        total = tag_total[tag]
        errors = tag_mistakes[tag]
        error_rate = errors / total if total else 0
        tag_error_stats.append((tag, total, errors, error_rate))

    tag_error_stats.sort(key=lambda x: x[3], reverse=True)  # sort by error rate descending

    for tag, total, errors, error_rate in tag_error_stats:
        print(f"Tag '{tag}': {errors} errors out of {total} ({error_rate:.2%} misclassification rate)")

    print(f"\nTotal time: {time.time() - _time:.2f} seconds")
        # --- Per-predicted-tag (misidentification) analysis ---
    print("\n=== Predicted Tag Misidentification Rates ===")
    pred_total = defaultdict(int)
    pred_wrong = defaultdict(int)

    for true_tag, pred_tag in zip(all_true_tags, all_pred_tags):
        pred_total[pred_tag] += 1
        if true_tag != pred_tag:
            pred_wrong[pred_tag] += 1

    pred_error_stats = []
    for tag in sorted(pred_total):
        total = pred_total[tag]
        wrong = pred_wrong[tag]
        misrate = wrong / total if total else 0
        pred_error_stats.append((tag, total, wrong, misrate))

    pred_error_stats.sort(key=lambda x: x[3], reverse=True)  # sort by highest error rate

    for tag, total, wrong, misrate in pred_error_stats:
        print(f"When predicted as '{tag}': {wrong} wrong out of {total} ({misrate:.2%} incorrect)")


if __name__ == '__main__':
    main()
