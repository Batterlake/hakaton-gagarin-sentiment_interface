import evaluate


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
