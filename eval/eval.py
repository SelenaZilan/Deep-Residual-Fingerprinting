import os
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest
import numpy as np
from dataset import HierarchicalFlowDataset
from model import AutoregressiveFlowTransformer
from config import (
    ATTACK_DAY_MAP,
    ATTACK_NAME_MAP,
    BATCH_SIZE,
    CHECKPOINTS,
    DEVICE as DEVICE_CONFIG,
    MODEL_PATH,
    MONDAY_CALIB_LIMIT,
    MONDAY_TRAIN_LIMIT,
    TARGET_E2E_FPR,
    TEST_DIRS,
    TRAIN_DIR,
)
from utils import build_prefix_dataset, load_flows_from_dir, add_flows_original_len, apply_thresholds_to_scores, get_nearest_checkpoint, record_earliest_alert, print_comparison_table
from residual_features import get_standardization_stats, extract_12d_features
from e2e_calibration import calibrate_e2e_thresholds_from_scores

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if DEVICE_CONFIG == "auto" else torch.device(DEVICE_CONFIG)

def fit_checkpoint_detectors(model, monday_flows, device):
    detectors = {}
    monday_flows = monday_flows[:MONDAY_TRAIN_LIMIT]

    print("\n--- Checkpoint-based iForest fitting on Monday benign flows---")
    for checkpoint in CHECKPOINTS:
        prefix_flows, _ = build_prefix_dataset(monday_flows, checkpoint)
        loader = DataLoader(HierarchicalFlowDataset(prefix_flows), batch_size=BATCH_SIZE, shuffle=False)
        stats = get_standardization_stats(model, loader, device, desc=f"Prefix {checkpoint} Standardization Statistics")
        train_x, _ = extract_12d_features(model, loader, stats, device, desc=f"Prefix {checkpoint} Features")
        clf = IsolationForest(n_estimators=150, n_jobs=-1, random_state=42).fit(train_x)
        detectors[checkpoint] = {
            "stats": stats,
            "clf": clf,
            "sample_count": len(prefix_flows),
        }
        print(
            f"Checkpoint {checkpoint:>2} detector fitted "
        )
    return detectors

def evaluate_checkpoint(model, flows, checkpoint, detector, device, test_name, batch_size):
    prefix_flows, orig_indices = build_prefix_dataset(flows, checkpoint)
    if len(prefix_flows) == 0:
        return None
    loader = DataLoader(HierarchicalFlowDataset(prefix_flows), batch_size=batch_size, shuffle=False)
    x, y = extract_12d_features(
        model,
        loader,
        detector["stats"],
        device,
        desc=f"{test_name} Prefix {checkpoint}"
    )
    scores = -detector["clf"].score_samples(x)

    return {
        "orig_indices": orig_indices,
        "labels": y,
        "scores": scores,
        "decision_steps": np.full(len(scores), checkpoint, dtype=np.int32),
        "threshold_refs": np.full(len(scores), checkpoint, dtype=np.int32)
    }
    
def evaluate_terminal(model, flows, detectors, device, test_name, checkpoint_steps, batch_size):
    grouped_indices = {}
    for idx, flow in enumerate(flows):
        terminal_len = len(flow["packets"])
        if terminal_len < 2 or terminal_len in checkpoint_steps:
            continue

        anchor_checkpoint = get_nearest_checkpoint(terminal_len, checkpoint_steps)
        grouped_indices.setdefault(anchor_checkpoint, []).append((idx, terminal_len))

    all_indices = []
    all_labels = []
    all_scores = []
    all_steps = []
    all_threshold_refs = []

    for anchor_checkpoint, indexed_lengths in sorted(grouped_indices.items()):
        selected_indices = [idx for idx, _ in indexed_lengths]
        terminal_lens = np.array([terminal_len for _, terminal_len in indexed_lengths], dtype=np.int32)
        orig_indices = np.array(selected_indices, dtype=np.int32)
        same_anchor_flows = [flows[index] for index in selected_indices]
        detector = detectors[anchor_checkpoint]
        loader = DataLoader(HierarchicalFlowDataset(same_anchor_flows), batch_size=batch_size, shuffle=False)
        x, y = extract_12d_features(
            model,
            loader,
            detector["stats"],
            device,
            desc=f"{test_name} Terminal -> CP{anchor_checkpoint}"
        )
        scores = -detector["clf"].score_samples(x)

        all_indices.append(orig_indices)
        all_labels.append(y)
        all_scores.append(scores)
        all_steps.append(terminal_lens)
        all_threshold_refs.append(np.full(len(scores), anchor_checkpoint, dtype=np.int32))

    return {
        "orig_indices": np.concatenate(all_indices),
        "labels": np.concatenate(all_labels),
        "scores": np.concatenate(all_scores),
        "decision_steps": np.concatenate(all_steps),
        "threshold_refs": np.concatenate(all_threshold_refs),
    }
    

def collect_checkpoint_score_results(model, flows, detectors, device, test_name):
    score_results = {}
    for checkpoint in CHECKPOINTS:
        score_results[checkpoint] = evaluate_checkpoint(
            model,
            flows,
            checkpoint,
            detectors[checkpoint],
            device,
            test_name,
            BATCH_SIZE
        )
    terminal_result = evaluate_terminal(model, flows, detectors, device, test_name, CHECKPOINTS, BATCH_SIZE)
    score_results["terminal"] = terminal_result

    return score_results

def evaluate_dataset(model, test_dir, detectors, global_thresholds):
    test_name = os.path.basename(test_dir)
    test_key = test_name.split("_maxpkts", 1)[0]
    attack = ATTACK_NAME_MAP.get(test_key, test_key)
    print(f"\n{'=' * 60}\n Evaluating: {test_key}")

    test_flows = load_flows_from_dir(test_dir, balance=(test_key in {"Wednesday-workingHours"}))
    add_flows_original_len(test_flows, test_name)     
    checkpoint_score_results = collect_checkpoint_score_results(
        model, test_flows, detectors, DEVICE, test_name,
    )
    global_results = apply_thresholds_to_scores(checkpoint_score_results, global_thresholds)
    all_summary = record_earliest_alert(test_flows, global_results)
    
    return {
        "attack": attack,
        "day": ATTACK_DAY_MAP.get(attack, "?"),
        "summary": all_summary,
    }


def main():
    model = AutoregressiveFlowTransformer().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    monday_all_benign = load_flows_from_dir(TRAIN_DIR, only_benign=True)
    monday_train_flows = monday_all_benign[:MONDAY_TRAIN_LIMIT]
    monday_calib_flows = monday_all_benign[MONDAY_TRAIN_LIMIT:MONDAY_TRAIN_LIMIT + MONDAY_CALIB_LIMIT]
    if len(monday_calib_flows) == 0:
        monday_calib_flows = monday_all_benign[:MONDAY_CALIB_LIMIT]

    detectors = fit_checkpoint_detectors(model, monday_train_flows, DEVICE)
    
    benign_score_results = collect_checkpoint_score_results(
        model,
        monday_calib_flows,
        detectors,
        DEVICE,
        test_name="Monday Benign E2E Calib",
    )
    
    global_calibration = calibrate_e2e_thresholds_from_scores(monday_calib_flows, benign_score_results, TARGET_E2E_FPR, CHECKPOINTS)
    global_thresholds = global_calibration["threshold_map"]
    
    print(
        f"Global E2E calibration | Shared percentile={global_calibration['shared_percentile']:.4f} | "
        f"Achieved benign FPR={global_calibration['achieved_fpr']:.4f}"
    )
    
    print("\n--- Evaluating test flows from Tuesday to Friday---")
    results = []
    for test_dir in TEST_DIRS:
        r = evaluate_dataset(model, test_dir, detectors, global_thresholds)
        results.append(r)
    print_comparison_table(results)

if __name__ == "__main__":
    main()

