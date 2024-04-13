import evaluate
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from tqdm.notebook import tqdm
from transformers import T5Tokenizer

from .model import NERModel


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        input_text = item["prefix"] + ": " + item["input_text"]
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        encoding["input_ids"] = encoding["input_ids"].squeeze(0)
        return encoding


def generate_answer_batched_v2(
    trained_model: NERModel,
    tokenizer: T5Tokenizer,
    data: pd.DataFrame,
    batch_size: int = 64,
    dataloader_workers: int = 16
):
    predictions = []
    dataset = CustomDataset(data, tokenizer, max_length=396)
    sequential_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sequential_sampler, num_workers=dataloader_workers)
    with torch.no_grad():
        for source_encoding in tqdm(dataloader):
            t0 = time.time()
            generated_ids = trained_model.model.generate(
                input_ids=source_encoding["input_ids"].cuda(),
                attention_mask=source_encoding["attention_mask"].cuda(),
                num_beams=3,
                max_length=80,
                temperature=0.5,
                repetition_penalty=1.0,
                early_stopping=True,
                use_cache=True,
            ).cpu()
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.append(preds)

    return sum(predictions, [])


def generate_answer_batched(
    trained_model: NERModel,
    tokenizer: T5Tokenizer,
    data: pd.DataFrame,
    batch_size: int = 64,
):
    predictions = []
    with torch.no_grad():
        for name, batch in tqdm(data.groupby(np.arange(len(data)) // batch_size)):
            source_encoding = tokenizer(
                (batch["prefix"] + ": " + batch["input_text"]).tolist(),
                max_length=396,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            generated_ids = trained_model.model.generate(
                input_ids=source_encoding["input_ids"].cuda(),
                attention_mask=source_encoding["attention_mask"].cuda(),
                num_beams=3,
                max_length=80,
                repetition_penalty=1.0,
                early_stopping=True,
                use_cache=True,
            ).cpu()

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.append(preds)

    return sum(predictions, [])


def evaluate_f1(predictions, labels):
    f1_metric = evaluate.load("f1")
    f1_score = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="weighted",
    )
    return f1_score["f1"]


def evaluate_accuracy(predictions, labels):
    acc_metric = evaluate.load("accuracy")
    acc_score = acc_metric.compute(
        predictions=predictions,
        references=labels,
    )
    return acc_score["accuracy"]


def evaluate_metric(
    company_labels, company_predictions, sentiment_labels, sentiment_predictions
):
    f1_score = evaluate_f1(company_predictions, company_labels)
    acc_score = evaluate_accuracy(sentiment_predictions, sentiment_labels)
    results = {"total": 100 * (f1_score + acc_score) / 2}
    results["f1"] = f1_score
    results["accuracy"] = acc_score
    return results
