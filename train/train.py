import os
import glob
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from dataset import HierarchicalFlowDataset
from model import AutoregressiveFlowTransformer
from validation import compute_iforest_validation_auc, format_iforest_validation_metrics

TRAIN_DIR = "../preprocessed/Monday-WorkingHours_maxpkts50_payload216" 
VALID_DIR = "../preprocessed/Friday-WorkingHours-Afternoon-DDos_maxpkts50_payload216"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = [1.0, 0.5, 0.2]               
TRAIN_MODE = True
eval_model_path = "../checkpoint/model.pth"

def compute_multi_step_loss_train(pf, pi, pl, tf, ti, tl, mask, log_vars, pred_steps=3):
    B, Seq, _ = tf.shape
    precisions = torch.exp(-log_vars)
    step_mses = []
    for s in range(pred_steps):
        shft = s + 1
        lf = F.mse_loss(pf[:, :-shft, s, :], tf[:, shft:, :], reduction='none').mean(dim=-1)
        li = F.mse_loss(pi[:, :-shft, s], ti[:, shft:], reduction='none')
        ll = F.mse_loss(pl[:, :-shft, s], tl[:, shft:], reduction='none')
        combined = (precisions[0] * lf + log_vars[0]) + (precisions[1] * li + log_vars[1]) + (precisions[2] * ll + log_vars[2])
        pad = (Seq - 1) - combined.size(1)
        if pad > 0: combined = F.pad(combined, (0, pad), value=-1e9)
        step_mses.append(combined.unsqueeze(-1) * WEIGHTS[s])
    stack = torch.cat(step_mses, dim=-1)
    final = (stack * stack.gt(-1e7).float()).sum(dim=-1) / stack.gt(-1e7).float().sum(dim=-1).clamp(min=1e-9)
    valid = (~mask[:, 1:]).float()
    return torch.sum(final * valid, dim=1) / valid.sum(dim=1).clamp(min=1e-9)

def load_flows_from_dir(d, only_benign=False):
    f = []
    for file in glob.glob(os.path.join(d, "*.pkl")):
        with open(file, 'rb') as fd:
            c = pickle.load(fd)
            f.extend([fl for fl in c if fl['label'].strip() == 'BENIGN'] if only_benign else c)
    return f

def main():
    print(f"Device: {DEVICE}")
    model = AutoregressiveFlowTransformer().to(DEVICE)
    
    print("Preloading training and validation data...")
    monday_flows = load_flows_from_dir(TRAIN_DIR, only_benign=True)
    valid_flows = load_flows_from_dir(VALID_DIR)

    if TRAIN_MODE:
        best_auc = None
        train_loader = DataLoader(HierarchicalFlowDataset(monday_flows), batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        total_steps = len(train_loader) * EPOCHS
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=LEARNING_RATE, 
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos', 
            final_div_factor=1e4
        )

        for epoch in range(1, EPOCHS + 1):
            # --- Stage 1: training ---
            model.train()
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
            for batch in train_pbar:
                for k, v in batch.items(): 
                    if isinstance(v, torch.Tensor): batch[k] = v.to(DEVICE)
                
                optimizer.zero_grad()
                pf, pi, pl, tf = model(batch)
                loss = compute_multi_step_loss_train(
                    pf, pi, pl, tf, 
                    batch['iats'], batch['orig_lens'], 
                    batch['attention_mask'], model.log_vars
                ).mean()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

            torch.save(model.state_dict(), f"../checkpoint/epoch{epoch}.pth")
            print(f"\nSaved latest model checkpoint to checkpoint/epoch{epoch}.pth")

            # --- Stage 2: validation after each epoch ---
            print(f"Epoch {epoch} finished. Running validation...")
            model.eval()
            valid_metrics = compute_iforest_validation_auc(
                model,
                monday_flows,
                valid_flows,
                DEVICE,
                BATCH_SIZE,
            )
            print(format_iforest_validation_metrics(valid_metrics))
            if valid_metrics["auc"] is not None and (best_auc is None or valid_metrics["auc"] > best_auc):
                best_auc = valid_metrics["auc"]
                torch.save(model.state_dict(), "../checkpoint/best.pth")
                print(f"Saved best model checkpoint to checkpoint/best.pth (AUC={best_auc:.4f})")
           
    else:
        print(f"Loading model checkpoint from {eval_model_path}")
        model.load_state_dict(torch.load(eval_model_path, map_location=DEVICE))
        model.eval()
        valid_metrics = compute_iforest_validation_auc(
            model,
            monday_flows,
            valid_flows,
            DEVICE,
            BATCH_SIZE,
        )
        print(format_iforest_validation_metrics(valid_metrics))


if __name__ == "__main__":
    main()