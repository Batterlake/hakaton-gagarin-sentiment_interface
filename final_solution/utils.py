import re

import numpy as np
import pandas as pd
import torch
import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


def format_predictions(predictions):
    predictions = re.sub(r"[^0-9;-]", "", predictions)
    pattern = r"\d+-\d+"
    matches = re.findall(pattern, predictions)
    if not matches:
        matches = ["1-3"]
    return ";".join(matches)


def postprocess_predictions(predictions: list[str]) -> list[list[tuple[int, float]]]:
    results = []
    for pr in predictions:
        sample = []
        ent = format_predictions(pr).split(";")
        for el in ent:
            splt = el.split("-")
            if len(splt) != 2:
                continue
            assert len(splt) == 2, pr
            try:
                sample.append((int(splt[0]), float(splt[1])))
            except ValueError:
                # unable to cast prediction to int or float
                pass
        results.append(list(sorted(map(list, set(sample)), key=lambda x: x[0])))
    return results


def generate_answer_batched(
    trained_model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    data: pd.DataFrame,
    batch_size: int = 64,
    num_beams: int = 3,
    max_source_length: int = 396,
    max_target_length: int = 80,
    verbose: bool = True,
):
    predictions = []
    with torch.no_grad():
        for name, batch in tqdm.tqdm(data.groupby(np.arange(len(data)) // batch_size)):
            try:
                source_encoding = tokenizer(
                    (batch["prefix"] + ": " + batch["input_text"]).tolist(),
                    max_length=max_source_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )

                generated_ids = trained_model.generate(
                    input_ids=source_encoding["input_ids"].cuda(),
                    attention_mask=source_encoding["attention_mask"].cuda(),
                    num_beams=num_beams,
                    max_length=max_target_length,
                    repetition_penalty=1.0,
                    early_stopping=True,
                    use_cache=True,
                ).cpu()

                preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                predictions.append(preds)
            except Exception as ex:
                if verbose:
                    print(ex)
                predictions.append(["nan" for _ in range(len(batch))])

    return sum(predictions, [])


def get_inference_model(path: str):
    return T5ForConditionalGeneration.from_pretrained(
        path, return_dict=True, torch_dtype=torch.float16
    )
