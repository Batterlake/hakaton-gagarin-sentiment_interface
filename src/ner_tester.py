from abc import abstractmethod


class BaseNERTester:

    @abstractmethod
    def infer_batch(self, input_texts: list[str]) -> list[set[int]]:
        """
        Method to run NER inference

        args:
        - input_texts: list[str] - list of input texts

        returns:
        list[set[int]] - list of sets of recognized entities
        """
        pass


def run_test(tester: BaseNERTester, path_to_df: str, batch_size: int) -> dict[str, float]:
    pass