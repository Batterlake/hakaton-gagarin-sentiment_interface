import evaluate
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer

from .model import BERTClassificationModel


def generate_answer_batched(
    trained_model: BERTClassificationModel,
    tokenizer: BertTokenizer,
    data: pd.DataFrame,
    batch_size: int = 64,
    max_length: int = 256,
):
    issuer_preds = []
    sentiment_preds = []
    trained_model.eval()
    with torch.no_grad():
        for _, batch in tqdm(data.groupby(np.arange(len(data)) // batch_size)):
            source_encoding = tokenizer(
                batch["MessageText"].tolist(),
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            issuer_output, sentiment_output = trained_model.forward(
                input_ids=source_encoding["input_ids"].cuda(),
                attention_mask=source_encoding["attention_mask"].cuda(),
            )
            issuer_output, sentiment_output = (
                issuer_output.cpu(),
                sentiment_output.cpu(),
            )

            sentiment_pred = sentiment_output.view(
                sentiment_output.shape[0], 5, -1
            ).argmax(1)
            issuer_ids = issuer_output.argmax(1).tolist()
            issuer_preds.append(issuer_output.argmax(1).tolist())
            sentiment_preds.append(
                [sentiment_pred[i, idx] for i, idx in enumerate(issuer_ids)]
            )

    return sum(issuer_preds, []), sum(sentiment_preds, [])


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
