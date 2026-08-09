import numpy as np

from tb_detection.evaluate import compute_metrics


def test_perfect_classifier_scores_1_everywhere():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.7, 0.9])
    m = compute_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["recall_tb_sensitivity"] == 1.0
    assert m["specificity_normal"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 2}


def test_threshold_behaviour():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.4, 0.6, 0.6, 0.4])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    # one FP (0.6 on a normal), one FN (0.4 on TB)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["specificity_normal"] == 0.5
