import sys
import typing as tp

sys.path.append("src")
import evaluate
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5Tokenizer

# sys.path.append("/home/worker/workspace/hakatons/hakaton-gagarin-sentiment_interface/src")
from src.t5.model import NERModel
from src.t5.utils import evaluate_metric, generate_answer_batched

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[EntityScoreType]  # list of entity scores,
#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]


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

m_name = "cointegrated/rut5-small"
tokenizer = T5Tokenizer.from_pretrained(m_name)
model = torch.load('final_model.pth')

def score_texts(
    messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    """

    df = pd.DataFrame({"message": messages})

    entities_found = generate_answer_batched(
        trained_model=model, tokenizer=tokenizer, data=df, batch_size=64
    )

    results = []

    for row in entities_found:
        for entity in row.split(";"):
            t = []
            tup = entity.split('-')
            entity_id, entity_score = tup
            t.append((entity_id, entity_score))
        results.append(t)

    return results
