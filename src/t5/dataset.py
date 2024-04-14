import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer


class NERDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_length: int = 396,
        target_max_token_length: int = 32,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_length = source_max_token_length
        self.target_max_token_length = target_max_token_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        source_encoding = self.tokenizer(
            data_row["prefix"] + ": " + data_row["input_text"],
            max_length=self.source_max_token_length,
            padding="max_length",
            truncation=True,
            # truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            input_text=data_row["prefix"] + ": " + data_row["input_text"],
            target_text=data_row["target_text"],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
        )


class NERDataModel(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        source_max_token_length=396,
        target_max_token_length=32,
        num_workers: int = 16,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.train_dataset = None
        self.test_dataset = None
        self.tokenizer = tokenizer
        self.source_max_token_length = source_max_token_length
        self.target_max_token_length = target_max_token_length
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = NERDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_length,
            self.target_max_token_length,
        )

        self.test_dataset = NERDataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_length,
            self.target_max_token_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)
