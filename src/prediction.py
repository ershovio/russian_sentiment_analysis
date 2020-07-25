from typing import Union, List

import torch
from pandas import Series
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)


class Predictor:
    def __init__(
            self,
            model: BertForSequenceClassification,
            tokenizer: BertTokenizer,
            max_length: int
    ):
        """
        :param model: pre trained `BertForSequenceClassification` model
        :param tokenizer: tokenizer for the model
        :param max_length: maximum tokens in a sequence for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._set_up_device()
        model.to(self.device)

    def predict(self, data: Union[Series, List[str]]) -> List[List[float]]:
        """
        Predict sentiments for given data
        :param data: input data
        :return: list of the same size as `data`
        where each item is a list with 3 logits
        for negative, neutral, and positive types
        """
        dataset = self._create_tensor_dataset(data)
        data_loader = DataLoader(
            dataset,
            batch_size=32
        )
        all_logits = []
        num_of_batches = len(data_loader)
        for batch_n, batch in enumerate(data_loader):
            print(f"Prediction is running for {batch_n + 1} batch / {num_of_batches}")
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            with torch.no_grad():
                logits, = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask
                )
            logits = logits.detach().cpu().tolist()
            all_logits += logits
        return all_logits

    def _create_tensor_dataset(
            self,
            data: Union[Series, List[str]]
    ):
        input_ids = []
        attention_masks = []

        for t in data:
            encoding = self.tokenizer.encode_plus(
                t,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            input_ids.append(encoding['input_ids'])
            attention_masks.append(encoding['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return TensorDataset(input_ids, attention_masks)

    def _set_up_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
