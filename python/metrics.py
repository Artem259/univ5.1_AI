import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix


# def recall_m(y_true, y_pred):  # TODO
#     y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=classes_num)
#     if len(y_pred.shape) == 1:
#         y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=classes_num)
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
#
# def precision_m(y_true, y_pred):
#     y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=classes_num)
#     if len(y_pred.shape) == 1:
#         y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=classes_num)
#     y_true = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

def recall_m(y_true, y_pred):
    # One-hot encoding for y_true and y_pred
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=classes_num)
    if len(y_pred.shape) == 1:
        y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=classes_num)

    # Cast to float32 for calculation
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate true positives and possible positives (i.e., actual positives)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    # Recall = TP / (TP + FN)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    # One-hot encoding for y_true and y_pred
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=classes_num)
    if len(y_pred.shape) == 1:
        y_pred = tf.one_hot(tf.cast(y_pred, tf.int32), depth=classes_num)

    # Cast to float32 for calculation
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate true positives and predicted positives (i.e., predicted positives)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    # Precision = TP / (TP + FP)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def model_predict(model, data):
    y_true = []
    y_pred = []
    for x_batch, y_batch in data:
        y_pred_batch = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(y_pred_batch, axis=1))
    return y_true, y_pred


def evaluate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    val_f1 = f1_m(y_true, y_pred).numpy()
    val_precision = precision_m(y_true, y_pred).numpy()
    val_recall = recall_m(y_true, y_pred).numpy()
    return {
        'f1': float(val_f1),
        'precision': float(val_precision),
        'recall': float(val_recall),
    }


def evaluate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm


def load_metrics(metrics_file):
    if metrics_file.is_file():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {}


def save_metrics(fold_metrics, metrics_file, cv_fold):
    metrics = load_metrics(metrics_file)
    metrics.pop('summary', None)
    metrics[f"cv{cv_fold}"] = fold_metrics
    metrics = dict(sorted(metrics.items(), key=lambda x: x[0]))

    num_folds = len(metrics)
    if num_folds > 0:
        summary_metrics = {
            'f1_mean': np.mean([m['f1'] for m in metrics.values()]),
            'f1_max': np.max([m['f1'] for m in metrics.values()]),
            'f1_min': np.min([m['f1'] for m in metrics.values()]),
            'precision_mean': np.mean([m['precision'] for m in metrics.values()]),
            'precision_max': np.max([m['precision'] for m in metrics.values()]),
            'precision_min': np.min([m['precision'] for m in metrics.values()]),
            'recall_mean': np.mean([m['recall'] for m in metrics.values()]),
            'recall_max': np.max([m['recall'] for m in metrics.values()]),
            'recall_min': np.min([m['recall'] for m in metrics.values()]),
        }
        metrics['summary'] = summary_metrics

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)


classes_num = None
