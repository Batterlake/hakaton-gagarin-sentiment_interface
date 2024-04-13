import evaluate
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
from transformers import T5Tokenizer

from .model import NERModel


def generate_answer_batched(
    trained_model: NERModel,
    tokenizer: T5Tokenizer,
    data: pd.DataFrame,
    batch_size: int = 64,
    n_beams: int = 3,
    max_length: int = 396,
):
    predictions = []
    with torch.no_grad():
        for name, batch in tqdm(data.groupby(np.arange(len(data)) // batch_size)):
            source_encoding = tokenizer(
                (batch["prefix"] + ": " + batch["input_text"]).tolist(),
                max_length=max_length,
                padding="longest",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            generated_ids = trained_model.generate(
                input_ids=source_encoding["input_ids"].cuda(),
                attention_mask=source_encoding["attention_mask"].cuda(),
                num_beams=n_beams,
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
