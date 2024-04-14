import typing as tp

import pandas as pd
from transformers import T5Tokenizer

from .utils import generate_answer_batched, get_inference_model, postprocess_predictions

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[EntityScoreType]  # list of entity scores,


def initialize_model(
    t_name: str = "cointegrated/rut5-small", p_name: str = "./pretrained-rut5-2-fp16"
):
    tokenizer = T5Tokenizer.from_pretrained(t_name)
    model = get_inference_model(p_name).cuda()
    return model, tokenizer


def score_texts(
    messages: tp.Iterable[str],
    model,
    tokenizer,
    *args,
    **kwargs,
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

    if not messages:
        return [[tuple()]]

    df = pd.DataFrame({"input_text": messages})
    df["prefix"] = "clsorg"
    predictions = generate_answer_batched(
        trained_model=model,
        tokenizer=tokenizer,
        data=df,
        batch_size=64,
        num_beams=1,
        max_target_length=40,
    )
    results = postprocess_predictions(predictions)

    return results
