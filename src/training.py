import datetime
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
from pandas import DataFrame
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import (
    TensorDataset,
    DataLoader
)
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from transformers.optimization import AdamW

from .metrics import compute_metrics, compute_average_metrics


class Trainer:
    def __init__(
            self,
            model: BertForSequenceClassification,
            tokenizer: BertTokenizer,
            seed: int = 100
    ):
        """
        :param model: `BertForSequenceClassification` model to train, num_labels should be set to 3
        :param tokenizer: tokenizer for the model
        :param seed: seed for reproducible results
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._set_up_device()
        self.model = model
        model.cuda()
        model.to(self.device)
        self.tokenizer = tokenizer

    def train(
            self,
            epochs: int,
            train_df: DataFrame,
            eval_df: DataFrame,
            dir_to_save: Optional[str] = None,
            max_length: int = 16,
            batch_size: int = 16,
            report_frequency: int = 50
    ):
        """
        Train model
        :param epochs: number of epochs for training
        :param train_df: training data, should contain "text" and "label" columns,
        labels should be -1, 0 or 1
        :param eval_df: evaluation data, should contain "text" and "label" columns,
        labels should be -1, 0 or 1
        :param dir_to_save: directory for saving training model,
        if wasn't set the trained model wouldn't saved
        :param max_length: maximum tokens in a sequence for the model
        :param batch_size: train and eval batch size
        :param report_frequency: frequency to print loss and training time
        (they will be printed every `report_frequency` batch)
        """
        if dir_to_save:
            if not os.path.isdir(dir_to_save):
                raise ValueError(f"'{dir_to_save}' is not a directory")

        train_dataset = self._create_tensor_dataset(train_df, max_length=max_length)
        eval_dataset = self._create_tensor_dataset(eval_df, max_length=max_length)

        train_data_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size
        )
        eval_data_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=2e-5,
            eps=1e-8
        )
        total_steps = len(train_data_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        for epoch_i in range(epochs):
            print(f"======== Epoch {epoch_i + 1} / {epochs} ========")
            self._train_epoch(
                data_loader=train_data_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                report_frequency=report_frequency
            )
            self._validate_epoch(data_loader=eval_data_loader)
        if dir_to_save:
            self.model.save_pretrained(dir_to_save)

    def _train_epoch(
            self,
            data_loader: DataLoader,
            optimizer: AdamW,
            scheduler: LambdaLR,
            report_frequency: int
    ):
        initial_time = time.time()
        total_train_loss = 0
        self.model.train()
        self.model.to(self.device)
        for step, batch in enumerate(data_loader):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            optimizer.zero_grad()
            loss, logits = self.model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            self._report_loss_and_time(
                step=step + 1,
                num_of_batches=len(data_loader),
                initial_time=initial_time,
                loss=loss,
                frequency=report_frequency
            )
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(data_loader)
        training_time = self._format_time_delta(initial_time)
        print(f"Average training loss: {avg_train_loss:.2f}")
        print(f"Training epoch took: {training_time}")

    def _validate_epoch(self, data_loader: DataLoader):
        print("Running Validation...")
        self.model.eval()
        total_eval_loss = 0
        metrics_list = []
        for batch_n, batch in enumerate(data_loader):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            with torch.no_grad():
                loss, logits = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            predicted = np.argmax(logits, axis=1).flatten()
            label_ids = b_labels.to("cpu").numpy()
            metrics = compute_metrics(predicted=predicted, expected=label_ids)
            metrics_list.append(metrics)

        print(
            f"Metrics for eval dataset:\n"
            f"{self._describe_average_metrics(compute_average_metrics(metrics_list))}"
        )
        print(f"Validation Loss: {total_eval_loss / len(data_loader):.2f}")

    def _create_tensor_dataset(
            self,
            data: DataFrame,
            max_length: int
    ):
        input_ids = []
        attention_masks = []
        for t in data["text"]:
            encoding = self.tokenizer.encode_plus(
                t,
                add_special_tokens=True,
                max_length=max_length,
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
        labels = torch.tensor(data["label"].apply(lambda x: x + 1))
        return TensorDataset(input_ids, attention_masks, labels)

    def _set_up_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _report_loss_and_time(
            self,
            step: int,
            num_of_batches: int,
            initial_time: float,
            loss: float,
            frequency: int
    ):
        if step % frequency == 0 and not step == 0:
            elapsed = self._format_time_delta(initial_time)
            print(f"Batch: {step} of {num_of_batches}. Loss: {loss:0.2f}. Time: {elapsed}")

    @staticmethod
    def _format_time_delta(initial_time: float):
        seconds = int(time.time() - initial_time)
        return str(datetime.timedelta(seconds=seconds))

    @staticmethod
    def _describe_average_metrics(metrics: Dict[str, Dict[str, float]]):
        return f"Avg precision: {metrics['average']['precision']:0.2f}\n" \
               f"Avg recall: {metrics['average']['recall']:0.2f}\n" \
               f"Avg F1: {metrics['average']['f1']:0.2f}"


class TrainerFactory:
    @staticmethod
    def trainer_with_default_model() -> Trainer:
        """
        :return: `Trainer` with "DeepPavlov/rubert-base-cased-conversational"
        pre trained model and tokenizer
        """
        pre_trained_model_name = "DeepPavlov/rubert-base-cased-conversational"
        model = BertForSequenceClassification.from_pretrained(
            pre_trained_model_name,
            num_labels=3
        )
        tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
        return Trainer(model=model, tokenizer=tokenizer)
