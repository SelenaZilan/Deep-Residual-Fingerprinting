import glob
import os
import pickle
import numpy as np
import pandas as pd

from config import CHECKPOINTS

def get_flow_label(flow):
    return 0 if flow["label"].strip().upper() == "BENIGN" else 1
    
def get_original_len(flow):
    csv_len = flow.get("original_total_packets", None)
    eff_len = len(flow["packets"])
    if csv_len is None:
        return eff_len
    return max(csv_len, eff_len)


def get_nearest_checkpoint(step, checkpoint_steps):
    return min(checkpoint_steps, key=lambda checkpoint: (abs(checkpoint - step), checkpoint))


def build_prefix_dataset(flows, prefix_len):
    prefix_flows = []
    orig_indices = []

    for idx, flow in enumerate(flows):
        if len(flow["packets"]) < prefix_len:
            continue

        prefix_flow = dict(flow)
        prefix_flow["packets"] = flow["packets"][:prefix_len]
        prefix_flows.append(prefix_flow)
        orig_indices.append(idx)

    return prefix_flows, np.array(orig_indices, dtype=np.int32)

def load_flows_from_dir(d, only_benign=False, balance=False):
    flows = []
    for file in glob.glob(os.path.join(d, "*.pkl")):
        with open(file, "rb") as fd:
            chunk = pickle.load(fd)
            if only_benign:
                flows.extend([fl for fl in chunk if fl["label"].strip().upper() == "BENIGN"])
            else:
                flows.extend(chunk)
    if balance and not only_benign:
        benign = [f for f in flows if f["label"].strip().upper() == "BENIGN"]
        malicious = [f for f in flows if f["label"].strip().upper() != "BENIGN"]
        n = min(len(benign), len(malicious))
        rng = np.random.RandomState(42)
        if len(benign) > n:
            benign = [benign[i] for i in rng.choice(len(benign), n, replace=False)]
        if len(malicious) > n:
            malicious = [malicious[i] for i in rng.choice(len(malicious), n, replace=False)]
        flows = benign + malicious
        rng.shuffle(flows)
        # print(f"  Balanced: {n} benign + {n} malicious = {2*n} flows")
    return flows


def add_flows_original_len(flows, test_name):
    csv_name = f"{os.path.basename(test_name).split('_maxpkts', 1)[0]}.pcap_ISCX.csv"
    csv_path = os.path.join("../CIC-IDS-2017/GeneratedLabelledFlows", csv_name)
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, low_memory=False, encoding='latin-1')
    df.columns = df.columns.str.strip()
    df['_total_pkts'] = (pd.to_numeric(df['Total Fwd Packets'], errors='coerce')
                         + pd.to_numeric(df['Total Backward Packets'], errors='coerce'))
    lookup = {}
    for _, row in df.iterrows():
        try:
            sport = int(row['Source Port'])
            dport = int(row['Destination Port'])
            total = int(row['_total_pkts'])
        except (ValueError, TypeError):
            continue
        fid = f"{row['Source IP']}-{row['Destination IP']}-{sport}-{dport}"
        ts = str(row['Timestamp']).strip()
        lookup[(fid, ts)] = total

    matched, total = 0, 0
    for flow in flows:
        total += 1
        key = (flow['flow_id'], str(flow['timestamp']).strip())
        if key in lookup:
            flow['original_total_packets'] = lookup[key]
            matched += 1
    # print(f"  CSV lookup: {matched}/{total} flows matched")

    
def apply_thresholds_to_scores(score_results, threshold_map):
    thresholded_results = {}

    for checkpoint, result in score_results.items():
        if result is None:
            thresholded_results[checkpoint] = None
            continue
        
        sample_thresholds = np.array(
            [threshold_map[ref]["threshold"] for ref in result["threshold_refs"]],
            dtype=np.float64,
        )
        preds = (result["scores"] > sample_thresholds).astype(int)
        thresholded_results[checkpoint] = {
            **result,
            "preds": preds,
        }
    return thresholded_results


def record_earliest_alert(flows, checkpoint_results):
    effective_lengths = np.array([len(flow["packets"]) for flow in flows])
    eligible_indices = np.where(effective_lengths >= 2)[0]
    original_lengths = np.array([get_original_len(flow) for flow in flows])
    flow_labels = np.array([get_flow_label(flow) for flow in flows])
    first_alert = np.zeros(len(flows), dtype=np.int32)

    ordered_keys = [checkpoint for checkpoint in CHECKPOINTS if checkpoint in checkpoint_results]
    ordered_keys.append("terminal") # terminal is always the last checkpoint

    for checkpoint in ordered_keys:
        result = checkpoint_results[checkpoint]
        if result is None:
            continue

        hit_mask = result["preds"] == 1
        hit_indices = result["orig_indices"][hit_mask]
        hit_steps = result["decision_steps"][hit_mask]
        for idx, step in zip(hit_indices, hit_steps):
            if first_alert[idx] == 0:
                first_alert[idx] = step

    eval_labels = flow_labels[eligible_indices]
    eval_alerts = first_alert[eligible_indices]
    eval_preds = eval_alerts > 0

    tp = np.sum((eval_preds == 1) & (eval_labels == 1))
    fp = np.sum((eval_preds == 1) & (eval_labels == 0))
    tn = np.sum((eval_preds == 0) & (eval_labels == 0))
    fn = np.sum((eval_preds == 0) & (eval_labels == 1))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    if tp > 0:
        tp_mask = (eval_preds == 1) & (eval_labels == 1)
        tp_first_alert_steps = eval_alerts[tp_mask]
        tp_original_lengths = original_lengths[eligible_indices][tp_mask]
        avg_early_alert_step = float(np.mean(tp_first_alert_steps))
        savings = tp_original_lengths - tp_first_alert_steps
        saving_ratios = savings / tp_original_lengths
        early_alert_analysis = {
            "count": int(tp),
            "avg_alert_packet_index": avg_early_alert_step,
            "min_alert_packet_index": float(np.min(tp_first_alert_steps)),
            "max_alert_packet_index": float(np.max(tp_first_alert_steps)),
            "q25_alert_packet_index": float(np.percentile(tp_first_alert_steps, 25)),
            "q75_alert_packet_index": float(np.percentile(tp_first_alert_steps, 75)),
            "avg_flow_len": float(np.mean(tp_original_lengths)),
            "median_flow_len": float(np.median(tp_original_lengths)),
            "avg_saving_ratio": float(np.mean(saving_ratios))
        }
    else:
        avg_early_alert_step = None
        early_alert_analysis = None

    return {
        "eligible_flows": int(len(eligible_indices)),
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "avg_alert_packet_index": avg_early_alert_step,
        "early_detection_analysis": early_alert_analysis,
    }

def print_comparison_table(results):
    print(f"\n{'#' * 70}")
    print(f"Detection Performance")
    print(f"{'#' * 70}")
    print(f"  {'Attack':<14} {'Day':>4} {'F1':>7} {'TPR':>7} {'FPR':>7}")
    print(f"  {'-'*14} {'-'*4} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        s = r["summary"]
        print(f"  {r['attack']:<14} {r['day']:>4} {s['f1']:>7.3f} {s['recall']:>7.3f} {s['fpr']:>7.3f}")

    print(f"\n{'#' * 70}")
    print(f"Early Detection Savings, TP flows")
    print(f"{'#' * 70}")
    print(f"  {'Attack':<14} {'Avg Obs':>8} {'Q1-Q3':>8} {'Min-Max':>8} {'Avg Len':>8} {'Saving':>8}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        da = r["summary"].get("early_detection_analysis")
        if da is None:
            print(f"  {r['attack']:<14} {'N/A':>8}")
            continue
        q1q3 = f"{da['q25_alert_packet_index']:.0f}-{da['q75_alert_packet_index']:.0f}"
        minmax = f"{da['min_alert_packet_index']:.0f}-{da['max_alert_packet_index']:.0f}"
        print(
            f"  {r['attack']:<14} "
            f"{da['avg_alert_packet_index']:>8.1f} "
            f"{q1q3:>8} "
            f"{minmax:>8} "
            f"{da['avg_flow_len']:>8.1f} "
            f"{da['avg_saving_ratio']*100:>7.1f}%"
        )
