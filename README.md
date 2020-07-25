# Sequence classification model for sentiment analysis of Russian conversational text

## Project structure
### [notebooks](notebooks)
`.ipynb` files
* [csv_preprocessing.ipynb](notebooks/csv_preprocessing.ipynb) - pre processing initial `csv` files
* [eda.ipynb](notebooks/csv_preprocessing.ipynb) - exploratory data analysis
* [training.ipynb](notebooks/training.ipynb) - model training
* [evaluation.ipynb](notebooks/evaluation.ipynb) - model evaluation

### [src](src)
`.py` files
* [`training.py`](src/training.py) contains `Trainer` class to train [BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification)
* [`prediction.py`](src/prediction.py) contains `Predictor` class to predict sentiments from text data using pre trained model
* [`metrics.py`](src/metrics.py) contains functions to compute `precision`, `recall`, and `f1` scores
* [`evaluation.py`](src/evaluation.py) contains a function to evaluate pre trained model using test data
* [`input_data_preprocessing.py`](src/input_data_preprocessing.py) contains utility function to fix initial `csv` files

## Metrics

|           | Positive | Neutral | Negative | Average |
|-----------|----------|---------|----------|---------|
| Precision |   0.744  |   0.717 |   0.799  |   0.753 |
| Recall    |   0.811  |   0.623 |   0.833  |   0.756 |
| F1        |   0.776  |   0.667 |   0.816  |   0.753 |