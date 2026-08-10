import numpy as np

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    valid = (y_true >= 0) & (y_true < num_classes)
    np.add.at(cm, (y_true[valid], y_pred[valid]), 1)
    return cm

def segmentation_metrics(y_true, y_pred, num_classes, phase):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    iou = tp / np.maximum(tp + fp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    return {
        f"{phase}_mIoU": float(np.mean(iou)),
        f"{phase}_pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)),
        f"{phase}_macro_precision": float(np.mean(precision)),
        f"{phase}_macro_recall": float(np.mean(recall)),
        f"{phase}_macro_f1": float(np.mean(f1)),
        f"{phase}_per_class_iou": iou.tolist(),
    }
