import typing as tp

import pandas as pd
from transformers import T5Tokenizer

from src.t5.model import get_inference_model
from src.t5.utils import generate_answer_batched, postprocess_predictions

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
        trained_model=model, tokenizer=tokenizer, data=df, batch_size=64
    )
    results = postprocess_predictions(predictions)

    return results


if __name__ == "__main__":
    s = (
        "США вводят санкции против российских банков:  Московский кредитный банк Банк «Урасиб» #USBN МТС Банк #MTSS Банк «Санкт-Петербург» #BSPB Банк «Зенит» Ланта Банк Металлинвестбанк Банк «Приморье» СДМ-Банк Уральский банк реконструкции и развития Банк "
        "Левобережный"
        "  \xa0\xa0\xa0\xa0 — Frank Media  ⚠️🇺🇸🇷🇺#санкции #россия  Минфин США в новом пакете санкций против РФ вводит рестрикции против 22 физических лиц и 83 юрлиц.  Минфин США вводит санкции против предприятий металлургического сектора и горной промышленности РФ  Минфин США также вводит дополнительные санкции против предприятий ВПК РФ  \xa0\xa0\xa0\xa0\xa0 — ТАСС  💥🇷🇺#TCSG = +6%  Тинькофф также не вошел в список новых санкций\xa0 США  (Британия тоже не внесла)  #банки #санкции #россия"
    )

    print(score_texts([s]))
