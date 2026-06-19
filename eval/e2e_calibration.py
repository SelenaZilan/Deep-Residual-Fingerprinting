import numpy as np
from config import PERCENTILE_SEARCH_HIGH, PERCENTILE_SEARCH_LOW, PERCENTILE_SEARCH_STEPS
from utils import apply_thresholds_to_scores, record_earliest_alert

def build_threshold_map_from_percentile(score_results, percentile, checkpoint_steps):
    threshold_map = {}
    for checkpoint in checkpoint_steps:
        result = score_results.get(checkpoint)
        if result is None or len(result["scores"]) == 0:
            continue
        threshold_map[checkpoint] = {
            "threshold": float(np.percentile(result["scores"], percentile)),
            "sample_count": len(result["scores"]),
        }
    return threshold_map


def calibrate_e2e_thresholds_from_scores(
    benign_flows,
    benign_score_results,
    target_fpr,
    checkpoint_steps,
    search_low=PERCENTILE_SEARCH_LOW,
    search_high=PERCENTILE_SEARCH_HIGH,
    search_steps=PERCENTILE_SEARCH_STEPS,
):
    best_percentile = None
    best_threshold_map = None
    best_summary = None

    lo = search_low
    hi = search_high
    for _ in range(search_steps):
        mid = (lo + hi) / 2.0
        threshold_map = build_threshold_map_from_percentile(benign_score_results, mid, checkpoint_steps)
        thresholded_results = apply_thresholds_to_scores(benign_score_results, threshold_map)
        e2e_summary = record_earliest_alert(benign_flows, thresholded_results)
        e2e_fpr = e2e_summary["fpr"]

        if e2e_fpr <= target_fpr:
            best_percentile = mid
            best_threshold_map = threshold_map
            best_summary = e2e_summary
            hi = mid
        else:
            lo = mid

    if best_threshold_map is None:
        fallback_percentile = search_high
        fallback_threshold_map = build_threshold_map_from_percentile(benign_score_results, fallback_percentile)
        thresholded_results = apply_thresholds_to_scores(benign_score_results, fallback_threshold_map)
        fallback_summary = record_earliest_alert(benign_flows, thresholded_results)
        
        return {
            "shared_percentile": fallback_percentile,
            "target_fpr": target_fpr,
            "achieved_fpr": fallback_summary["fpr"],
            "threshold_map": fallback_threshold_map,
        }

    return {
        "shared_percentile": best_percentile,
        "target_fpr": target_fpr,
        "achieved_fpr": best_summary["fpr"],
        "threshold_map": best_threshold_map
    }


