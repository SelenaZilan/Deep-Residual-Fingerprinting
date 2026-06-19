import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset import HierarchicalFlowDataset
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from eval.residual_features import get_standardization_stats, extract_12d_features

def compute_iforest_validation_auc(model, reference_flows, valid_flows, device, batch_size):
    reference_loader = DataLoader(
        HierarchicalFlowDataset(reference_flows),
        batch_size=batch_size,
        shuffle=False,
    )
    stand_stats = get_standardization_stats(
        model,
        reference_loader,
        device,
        desc="Validation standardization stats",
    )
    reference_x, _ = extract_12d_features(
        model,
        reference_loader,
        stand_stats,
        device,
        desc="Validation iForest train features",
    )
    clf = IsolationForest(n_estimators=150, n_jobs=-1, random_state=42).fit(reference_x)

    valid_loader = DataLoader(
        HierarchicalFlowDataset(valid_flows),
        batch_size=batch_size,
        shuffle=False,
    )
    valid_x, valid_y = extract_12d_features(
        model,
        valid_loader,
        stand_stats,
        device,
        desc="Validation standardized residuals",
    )

    scores = -clf.score_samples(valid_x)
    result = {
        "mean_score": float(np.mean(scores)),
    }
    if np.any(valid_y == 0):
        result["benign_mean_score"] = float(np.mean(scores[valid_y == 0]))
    if np.any(valid_y == 1):
        result["attack_mean_score"] = float(np.mean(scores[valid_y == 1]))
    if len(np.unique(valid_y)) >= 2:
        result["auc"] = float(roc_auc_score(valid_y, scores))
    return result


def format_iforest_validation_metrics(metrics):
    if metrics is None:
        return "Validation iForest | N/A"

    auc_text = "N/A" if metrics["auc"] is None else f"{metrics['auc']:.4f}"
    benign_mean_score = (
        "N/A"
        if metrics["benign_mean_score"] is None
        else f"{metrics['benign_mean_score']:.4f}"
    )
    attack_mean_score = (
        "N/A"
        if metrics["attack_mean_score"] is None
        else f"{metrics['attack_mean_score']:.4f}"
    )
    return (
        "Validation iForest | "
        f"AUC={auc_text} | "
        f"mean_score={metrics['mean_score']:.4f} | "
        f"benign_mean={benign_mean_score} | "
        f"attack_mean={attack_mean_score}"
)
