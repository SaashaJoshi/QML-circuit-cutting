from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np


def encode_y(y):
    y_encoded = y.replace({0: -1, 1: 1})
    return y_encoded


def predict(reconstructed_expvals):
    y_predict = np.sign(reconstructed_expvals)
    return validate_predict(y_predict)


def validate_predict(y_hat: np.ndarray) -> np.ndarray:
    target_encoder = LabelEncoder()
    try:
        check_is_fitted(target_encoder)
        return target_encoder.inverse_transform(y_hat).squeeze()
    except NotFittedError:
        return y_hat


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
