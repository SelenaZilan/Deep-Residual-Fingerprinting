import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch

def get_single_dim_stats(p, t, attention_mask, is_feat=False):
    if is_feat:
        raw_mse = F.mse_loss(p[:, :-1, 0, :], t[:, 1:, :], reduction="none").mean(dim=-1)
    else:
        raw_mse = F.mse_loss(p[:, :-1, 0], t[:, 1:], reduction="none")

    valid_mask = (~attention_mask[:, 1:]).float()
    mse_masked = raw_mse * valid_mask
    count = valid_mask.sum(dim=1).clamp(min=1.0)

    mean_val = mse_masked.sum(dim=1) / count
    mean_sq = (mse_masked ** 2).sum(dim=1) / count
    std_val = torch.sqrt((mean_sq - mean_val ** 2).clamp(min=1e-9))
    max_val, _ = torch.max(raw_mse.masked_fill(valid_mask == 0, -1e9), dim=1)

    sorted_mse, _ = torch.sort(raw_mse.masked_fill(valid_mask == 0, -1e9), dim=1, descending=True)
    p90_idx = (count * 0.1).long()
    p90_val = torch.gather(sorted_mse, 1, p90_idx.unsqueeze(1)).squeeze(1)

    return torch.stack(
        [mean_val, std_val, max_val.clamp(min=0), p90_val.clamp(min=0)],
        dim=1
    ).detach().cpu().numpy()


@torch.no_grad()
def get_standardization_stats(model, loader, device, desc):
    model.eval()
    all_f, all_i, all_l = [], [], []

    for batch in tqdm(loader, desc=desc):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        pf, pi, pl, tf = model(batch)
        all_f.append(get_single_dim_stats(pf, tf, batch["attention_mask"], True))
        all_i.append(get_single_dim_stats(pi, batch["iats"], batch["attention_mask"]))
        all_l.append(get_single_dim_stats(pl, batch["orig_lens"], batch["attention_mask"]))

    return {
        "feat": (np.mean(np.vstack(all_f), axis=0), np.std(np.vstack(all_f), axis=0) + 1e-9),
        "iat": (np.mean(np.vstack(all_i), axis=0), np.std(np.vstack(all_i), axis=0) + 1e-9),
        "len": (np.mean(np.vstack(all_l), axis=0), np.std(np.vstack(all_l), axis=0) + 1e-9),
    }


@torch.no_grad()
def extract_12d_features(model, loader, stand_stats, device, desc):
    model.eval()
    all_x, all_y = [], []
    m_f, s_f = stand_stats["feat"]
    m_i, s_i = stand_stats["iat"]
    m_l, s_l = stand_stats["len"]

    for batch in tqdm(loader, desc=desc):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        pf, pi, pl, tf = model(batch)
        f_s = (get_single_dim_stats(pf, tf, batch["attention_mask"], True) - m_f) / s_f
        i_s = (get_single_dim_stats(pi, batch["iats"], batch["attention_mask"]) - m_i) / s_i
        l_s = (get_single_dim_stats(pl, batch["orig_lens"], batch["attention_mask"]) - m_l) / s_l

        f_s = np.sign(f_s) * np.log1p(np.abs(f_s))
        i_s = np.sign(i_s) * np.log1p(np.abs(i_s))
        l_s = np.sign(l_s) * np.log1p(np.abs(l_s))

        all_x.append(np.hstack([f_s, i_s, l_s]))
        all_y.extend(batch["label"].detach().cpu().numpy())

    return np.vstack(all_x), np.array(all_y)
