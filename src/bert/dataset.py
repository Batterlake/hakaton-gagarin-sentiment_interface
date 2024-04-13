from collections import namedtuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class BERTClassificationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_length: int = 256,
        num_issuers_classes: int = 250,
        num_sentiment_classes: int = 5,
        message_col_name: str = "MessageText",
        issuer_id_col_name: str = "issuerid",
        sentiment_score_col_name: str = "SentimentScore",
    ):
        Sample = namedtuple("Sample", ["message", "issuer_ids", "sentiments"])
        samples = []

        for message, group in df.groupby(message_col_name):
            samples.append(
                Sample(
                    message,
                    group[issuer_id_col_name].tolist(),
                    group[sentiment_score_col_name].tolist(),
                )
            )

        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_issuers_classes = num_issuers_classes
        self.num_sentiment_classes = num_sentiment_classes

    def __len__(self):
        return len(self.samples)

    def _one_hot(self, dim, positions):
        vec = torch.zeros(dim)
        for pos in positions:
            if isinstance(pos, int):
                vec[pos] = 1
            else:
                vec[pos[0], pos[1]] = 1

        return vec

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample.message,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        issuer_labels = self._one_hot(self.num_issuers_classes, sample.issuer_ids)
        sentiment_labels = self._one_hot(
            (self.num_sentiment_classes, self.num_issuers_classes),
            list(zip(sample.sentiments, sample.issuer_ids)),
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "issuer_labels": issuer_labels,
            "sentiment_labels": sentiment_labels,
        }


class BERTClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: BertTokenizer,
        batch_size: int = 8,
        max_length: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.train_dataset = None
        self.test_dataset = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kwargs = kwargs

    def setup(self, stage=None):
        self.train_dataset = BERTClassificationDataset(
            self.train_df,
            self.tokenizer,
            self.max_length,
            **self.kwargs,
        )

        self.test_dataset = BERTClassificationDataset(
            self.test_df,
            self.tokenizer,
            self.max_length,
            **self.kwargs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=16)
