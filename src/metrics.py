from typing import List, Dict, Union

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(
        predicted: Union[np.ndarray, List[int]],
        expected: Union[np.ndarray, List[int]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes precision, recall, and f1 scores
    :param predicted: predicted values, contains 0, 1, and 2 for negative, neutral, and positive
    :param expected: true values, contains 0, 1, and 2 for negative, neutral, and positive
    :return: Dict in the following format
        {
            "sentiment type":
                {
                    "score type": score_result
                }
        }
        where "sentiment type" is "negative", "neutral", "positive" or "average"
        and "score type" is "precision", "recall" or "f1"
    """
    assert len(predicted) == len(expected), \
        f"Number of items in predicted and expected should be equal, " \
        f"but {len(predicted)} and {len(expected)} given"
    for i in np.concatenate([predicted, expected]):
        assert i == 0 or i == 1 or i == 2, \
            f"Predicted and expected should contain only 0, 1 or 2, but {i} given," \
            f"\nPredicted: {predicted}\nExpected: {expected}"

    negative_predicted = [1 if i == 0 else 0 for i in predicted]
    negative_expected = [1 if i == 0 else 0 for i in expected]

    neutral_predicted = [1 if i == 1 else 0 for i in predicted]
    neutral_expected = [1 if i == 1 else 0 for i in expected]

    positive_predicted = [1 if i == 2 else 0 for i in predicted]
    positive_expected = [1 if i == 2 else 0 for i in expected]

    res = {
        "negative": {
            "precision": precision_score(negative_expected, negative_predicted),
            "recall": recall_score(negative_expected, negative_predicted),
            "f1": f1_score(negative_expected, negative_predicted)
        },
        "neutral": {
            "precision": precision_score(neutral_expected, neutral_predicted),
            "recall": recall_score(neutral_expected, neutral_predicted),
            "f1": f1_score(neutral_expected, neutral_predicted)
        },
        "positive": {
            "precision": precision_score(positive_expected, positive_predicted),
            "recall": recall_score(positive_expected, positive_predicted),
            "f1": f1_score(positive_expected, positive_predicted)
        },
        "average": {}
    }
    for metric in ["precision", "recall", "f1"]:
        res["average"][metric] = (res["negative"][metric] +
                                  res["neutral"][metric] +
                                  res["positive"][metric]) / 3
    return res


def compute_average_metrics(
        metrics: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes average metrics
    :param metrics: list of metrics computed by `compute_metrics`
    :return: Average metrics in the following format
        {
            "sentiment type":
                {
                    "score type": score_result
                }
        }
        where "sentiment type" is "negative", "neutral", "positive" or "average"
        and "score type" is "precision", "recall" or "f1"
    """
    number_of_metrics = len(metrics)
    res = {}
    for sentiment_type in ["negative", "neutral", "positive", "average"]:
        res[sentiment_type] = {}
        for metric in ["precision", "recall", "f1"]:
            list_of_metrics = [i[sentiment_type][metric] for i in metrics]
            res[sentiment_type][metric] = sum(list_of_metrics) / number_of_metrics
    return res


def pretty_print_metrics(metrics: Dict[str, Dict[str, float]]):
    """
    Prints metrics that was computed by `compute_metrics` in the following format:
    negative precision: 0.563
    neutral precision: 0.675
    ...
    """
    for sentiment_type in metrics.keys():
        for metric in metrics[sentiment_type].keys():
            metric_value = metrics[sentiment_type][metric]
            print(f"{sentiment_type} {metric}: {metric_value:0.3f}")
