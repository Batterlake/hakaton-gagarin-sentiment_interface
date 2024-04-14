import json
import pathlib
import time
import typing as tp

# PATH_TO_TEST_DATA = pathlib.Path("data") / "test_texts.json"
PATH_TO_TEST_DATA = pathlib.Path("data") / "test_texts_2.json"
PATH_TO_OUTPUT_DATA = pathlib.Path("results") / "output_scores.json"


def load_data(path: pathlib.PosixPath = PATH_TO_TEST_DATA) -> tp.List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_data(data, path: pathlib.PosixPath = PATH_TO_OUTPUT_DATA):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


class StupyLogger:
    def error(self, message: str):
        print(message)


def main():
    begin = time.perf_counter()
    texts = []
    model = None
    tokenizer = None
    scores = []
    try:
        from loguru import logger

        from final_solution.solution import initialize_model, score_texts

    except ImportError as err:
        logger = StupyLogger()
        logger.error(f"Exception at import: {err}")

    try:
        texts = load_data()
    except Exception as ex:
        logger.error(f"Unable to load data: {ex}")
    dt = time.perf_counter()
    try:
        model, tokenizer = initialize_model()
    except Exception as ex:
        logger.error(f"Unable to load model and/or tokenizer: {ex}")
    mdl = time.perf_counter()

    try:
        scores = score_texts(texts, model=model, tokenizer=tokenizer)
    except Exception as ex:
        logger.error(f"Unable to load model and/or tokenizer: {ex}")

    scr = time.perf_counter()
    try:
        save_data(scores)
    except Exception as ex:
        logger.error(f"Unable to save predictions: {ex}")
    sv = time.perf_counter()
    print(
        f"Load data: {dt-begin}, Load model: {mdl-dt}, Score: {scr-mdl}, Save: {sv-scr}"
    )


if __name__ == "__main__":
    main()
