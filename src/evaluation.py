import numpy as np
from pandas import DataFrame
from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

from .metrics import compute_metrics, pretty_print_metrics
from .prediction import Predictor


def evaluate(
        model: BertForSequenceClassification,
        tokenizer: BertTokenizer,
        data: DataFrame,
        max_length: int = 128,
):
    """
    Evaluate given model using given data
    and prints precision, recall, and F1 for
    negative, neutral, positive, and average cases
    :param model: model to evaluate
    :param tokenizer: model's tokenizer
    :param data: test data, should contains "text" and "label" columns.
    Labels should be -1, 0 or 1
    :param max_length: maximum tokens in a sequence for the BERT model
    """
    predictor = Predictor(model=model, tokenizer=tokenizer, max_length=max_length)
    predicted_logits = predictor.predict(data["text"])
    predicted_labels = np.argmax(predicted_logits, axis=1)
    expected_labels = data["label"].apply(lambda x: x + 1)
    metrics = compute_metrics(predicted=predicted_labels, expected=expected_labels)
    pretty_print_metrics(metrics)
