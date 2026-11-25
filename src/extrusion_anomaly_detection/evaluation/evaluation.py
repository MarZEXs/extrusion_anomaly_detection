import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


def evaluate(model, dataloader, device, threshold=None):
    fault_names = ("flow_rate", "feed_rate", "hotend")
    scores, faults = [], []

    model.eval()
    with torch.inference_mode():
        for image, parameters in tqdm(dataloader):
            image = image.to(device)
            scores.append(model.score(image).reshape(-1).cpu())

            batch_faults = torch.column_stack(
                [parameters[f"{name}_class"] != 1 for name in fault_names]
            )
            faults.append(batch_faults)

    scores = torch.cat(scores).numpy()
    faults = torch.cat(faults).numpy()
    labels = faults.any(axis=1).astype(int)

    metrics = {
        "auroc": roc_auc_score(labels, scores),
        "ap": average_precision_score(labels, scores),
    }

    for name, is_fault in zip(fault_names, faults.T):
        eval_mask = (labels == 0) | is_fault
        y_true = labels[eval_mask]
        y_score = scores[eval_mask]

        metrics[f"auroc_{name}"] = roc_auc_score(y_true, y_score)
        metrics[f"ap_{name}"] = average_precision_score(y_true, y_score)

    if threshold is not None:
        predictions = (scores >= threshold).astype(int)
        metrics.update(
            {
                "precision": precision_score(labels, predictions, zero_division=0),
                "recall": recall_score(labels, predictions, zero_division=0),
                "f1": f1_score(labels, predictions, zero_division=0),
            }
        )
    return metrics
