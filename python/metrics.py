import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix


def _cast_y(y_true, y_pred):
    if len(y_pred.shape) > 1:
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int64)
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.cast(y_pred, tf.int64)
    return y_true, y_pred


def precision_macro(y_true, y_pred):
    y_true, y_pred = _cast_y(y_true, y_pred)

    class_precisions = []
    for c in range(classes_num):
        true_positives = tf.reduce_sum(tf.cast((y_pred == c) & (y_true == c), tf.float32))
        predicted_positives = tf.reduce_sum(tf.cast(y_pred == c, tf.float32))

        precision = true_positives / (predicted_positives + K.epsilon())
        class_precisions.append(precision)
    return tf.reduce_mean(class_precisions)


def recall_macro(y_true, y_pred):
    y_true, y_pred = _cast_y(y_true, y_pred)

    class_recalls = []
    for c in range(classes_num):
        true_positives = tf.reduce_sum(tf.cast((y_pred == c) & (y_true == c), tf.float32))
        possible_positives = tf.reduce_sum(tf.cast(y_true == c, tf.float32))

        recall = true_positives / (possible_positives + K.epsilon())
        class_recalls.append(recall)
    return tf.reduce_mean(class_recalls)


def f1_macro(y_true, y_pred):
    y_true, y_pred = _cast_y(y_true, y_pred)

    class_f1_scores = []
    for c in range(classes_num):
        true_positives = tf.reduce_sum(tf.cast((y_pred == c) & (y_true == c), tf.float32))
        predicted_positives = tf.reduce_sum(tf.cast(y_pred == c, tf.float32))
        possible_positives = tf.reduce_sum(tf.cast(y_true == c, tf.float32))

        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())

        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        class_f1_scores.append(f1)
    return tf.reduce_mean(class_f1_scores)


def model_predict(model, data):
    y_true = []
    y_pred = []
    for x_batch, y_batch in data:
        y_pred_batch = model.predict(x_batch, verbose=0)
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(y_pred_batch, axis=1))
    return np.array(y_true), np.array(y_pred)


def evaluate_metrics(y_true, y_pred):
    val_f1 = f1_macro(y_true, y_pred).numpy()
    val_precision = precision_macro(y_true, y_pred).numpy()
    val_recall = recall_macro(y_true, y_pred).numpy()
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


classes_num = 0
