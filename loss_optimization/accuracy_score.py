def encode_y(y):
    y_encoded = y.replace({0: -1, 1: 1})
    return y_encoded


def get_accuracy_score(y_true, y_pred):
    encoded_y_true = encode_y(y_true)
    if len(encoded_y_true) != len(y_pred):
        raise ValueError("Input lists must have the same length.")

    correct_predictions = sum(
        1 for true, pred in zip(encoded_y_true, y_pred) if true == pred
    )
    total_instances = len(encoded_y_true)

    accuracy = correct_predictions / total_instances
    return accuracy
