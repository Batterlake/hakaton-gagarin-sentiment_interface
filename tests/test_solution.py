import pytest

import final_solution


def test_empty():
    """If score_texts do not pass this test, it is fine"""

    assert not bool(final_solution.solution.score_texts([]))
    
    # nothing was found
    nothing = final_solution.solution.score_texts([""])
    assert nothing == [[tuple()]]


def test_one_message():
    """Format of answers is important"""
    messages = ["Сбер, он и в Африке Сбер"]
    correct_scores = [[(150, 3.0)]]
    
    assert final_solution.solution.score_texts(messages) == correct_scores


def test_two_entities_one_message():
    """Order of companies inside one message is not important"""
    messages = ["Сбер, он и в Африке. Тинькофф, он и в Африке Тинькофф"]
    correct_scores = [[(150, 3.0), (225, 3.0)]]
    
    scores = final_solution.solution.score_texts(messages)

    assert [set(s) == set(cs) for s, cs in zip(scores, correct_scores)]


def test_two_entities_two_messages():
    """"""
    messages = ["Сбер, он и в Африке Сбер", "Тинькофф, он и в Африке Тинькофф"]
    correct_scores = [[(150, 3.0)], [(225, 3.0)]]

    assert final_solution.solution.score_texts(messages) == correct_scores


def test_large_sequence(N = 10 ** 3):
    """No matter how large N is, score_texts function should work"""
    message = "Сбер, он и в Африке Сбер"
    correct_score = [(150, 3.0)]

    assert final_solution.solution.score_texts([message] * N) == [correct_score] * N
