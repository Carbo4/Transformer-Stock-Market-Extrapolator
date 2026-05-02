import os
import glob
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from Model.Dataset import StockDataset
from Model.backbone import Backbone
from Model.heads import DriftHead, JumpHead
import visualize


class LazyStockDataset(torch.utils.data.Dataset):
    """Top-level lazy dataset to allow multiprocessing pickling.
    It creates per-window temporary StockDataset instances on demand to avoid
    precomputing all windows for a file.
    """
    def __init__(self, df, stride=8, window_len=192, ctx_len=128, max_windows=None):
        self.df = df.reset_index(drop=True)
        self.stride = stride
        self.window_len = window_len
        self.ctx_len = ctx_len
        self.max_windows = max_windows
        total = max(0, (len(self.df) - window_len) // stride + 1)
        if max_windows is not None:
            total = min(total, max_windows)
        self._len = total

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        offset = index * self.stride
        window = self.df.iloc[offset:offset + self.window_len]
        tmp = StockDataset(window, stride=1, window_len=self.window_len, ctx_len=self.ctx_len)
        return tmp[0]


class MOEBackbone(nn.Module):
    def __init__(self, encoder_layer_factory, num_layers, num_experts=2, d_model=64):
        super().__init__()
        self.experts = nn.ModuleList([
            Backbone(encoder_layer_factory(), num_layers)
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, src, t):
        # src: (B, T, E)
        # compute gate logits from time-pooled src
        pooled = src.mean(dim=1)  # (B, E)
        logits = self.gate(pooled)
        weights = torch.softmax(logits, dim=-1)  # (B, num_experts)

        expert_outs = [expert(src, t) for expert in self.experts]
        # each expert_out: (B, T, E)
        out = torch.zeros_like(expert_outs[0])
        for i, eo in enumerate(expert_outs):
            w = weights[:, i].unsqueeze(-1).unsqueeze(-1)
            out = out + eo * w
        return out


def read_pse_files(path='data/PSE'):
    files = sorted(glob.glob(os.path.join(path, '*.csv')))
    train_datasets = []
    train_names = []
    val_datasets = []
    val_names = []
    min_span_days = 180
    # use module-level LazyStockDataset to avoid pickling issues with workers

    for f in files:
        try:
            # read without forcing types. Files are DOHLCV in cols 0..5
            df = pd.read_csv(f, header=None, usecols=range(6), low_memory=False)
            if df.shape[1] < 6:
                continue
            df = df.iloc[:, :6]
            df.columns = ['D', 'O', 'H', 'L', 'C', 'V']
            # convert numeric columns to floats, coerce errors
            for col in ['O', 'H', 'L', 'C', 'V']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # If D isn't a datetime, synthesize a monotonic DatetimeIndex to save costly parsing
            try:
                df['D'] = pd.to_datetime(df['D'], errors='coerce')
                if df['D'].isna().all():
                    raise ValueError('empty dates')
            except Exception:
                df['D'] = pd.date_range('2000-01-01', periods=len(df), freq='T')

            if len(df) < 200:
                continue
            # require sufficient date span to make cyclical time features meaningful
            try:
                span_days = (df['D'].max() - df['D'].min()).days
            except Exception:
                span_days = 0
            if span_days < min_span_days:
                print('Skipping', f, 'span_days=', span_days)
                continue

            # Split by year: train = before 2024, val = 2024 and after
            df_train = df[df['D'].dt.year < 2024].reset_index(drop=True)
            df_val = df[df['D'].dt.year >= 2024].reset_index(drop=True)

            name = os.path.splitext(os.path.basename(f))[0]

            if len(df_train) >= 200:
                ds_train = LazyStockDataset(df_train, stride=8, window_len=192, ctx_len=128, max_windows=500)
                if len(ds_train) > 0:
                    train_datasets.append(ds_train)
                    train_names.append(name)
                    print('Loaded train', f, 'windows=', len(ds_train))

            if len(df_val) >= 64:
                ds_val = LazyStockDataset(df_val, stride=8, window_len=192, ctx_len=128, max_windows=500)
                if len(ds_val) > 0:
                    val_datasets.append(ds_val)
                    val_names.append(name)
                    print('Loaded val', f, 'windows=', len(ds_val))

        except Exception as e:
            print('Failed to read', f, e)
    return train_datasets, train_names, val_datasets, val_names


def collate_fn(batch):
    dts, xs, targets = zip(*batch)
    dts = torch.stack(dts)
    xs = torch.stack(xs)
    stacked = {}
    for k in targets[0].keys():
        stacked[k] = torch.stack([t[k] for t in targets])
    return dts, xs, stacked


def run(epochs=10, use_moe=False, num_experts=2, device=None, max_workers=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    train_datasets, train_names, val_datasets, val_names = read_pse_files('data/PSE')
    if not train_datasets:
        print('No training datasets (pre-2024) found in data/PSE')
        return
    # keep per-file datasets for validation visualizations (post-2023)
    concat = ConcatDataset(train_datasets)
    workers = max(0, (max_workers or min(4, (os.cpu_count() or 4) // 2)))
    loader = DataLoader(concat, batch_size=16, shuffle=True, collate_fn=collate_fn,
                        num_workers=workers, pin_memory=(device.type == 'cuda'))

    # model setup
    _, sample_x, sample_targets = train_datasets[0][0]
    n_features = sample_x.shape[1]
    d_model = 64
    nhead = 4

    def encoder_factory():
        return nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)

    if use_moe:
        backbone = MOEBackbone(encoder_factory, num_layers=3, num_experts=num_experts, d_model=d_model).to(device)
    else:
        backbone = Backbone(encoder_factory(), num_layers=3).to(device)

    input_proj = nn.Linear(n_features, d_model).to(device)
    drift = DriftHead(d_model=d_model, nhead=nhead, num_layers=2).to(device)
    jump = JumpHead(d_model=d_model, nhead=nhead, num_layers=2).to(device)

    opt = torch.optim.Adam(list(backbone.parameters()) + list(input_proj.parameters()) +
                           list(drift.parameters()) + list(jump.parameters()), lr=3e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    # regularization weight for predicted magnitudes (discourage large mags when mask prob is low)
    mag_reg_lambda = 0.1
    # weight for aligning predicted mask probability with future volume information
    vol_align_lambda = 0.3

    best_loss = float('inf')

    train_losses = []

    for ep in range(epochs):
        backbone.train()
        total_loss = 0.0
        n_steps = 0
        for dts, xs, targets in loader:
            dts = dts.to(device).float()
            xs = xs.to(device).float()
            tgt_smooth = targets['smooth_path'].to(device).float()
            tgt_volume_info = targets['volume_information'].to(device).float()
            tgt_sparse = targets['sparse_jumps'].to(device).float()

            emb = input_proj(xs)
            trunk = backbone(emb, dts)

            last_ctx = xs[:, -1, 0]
            if tgt_smooth.shape[1] == 0:
                continue
            tgt_in = torch.cat([last_ctx.unsqueeze(1), tgt_smooth[:, :-1]], dim=1)
            pred_mean, pred_logvar, pred_skew = drift(trunk, dts, tgt_in)
            # skew-normal NLL
            var = torch.exp(pred_logvar) + 1e-6
            sigma = torch.sqrt(var)
            z = (tgt_smooth.unsqueeze(-1) - pred_mean) / sigma
            std_normal = torch.distributions.Normal(0.0, 1.0)
            log_phi = -0.5 * (z ** 2) - 0.5 * math.log(2 * math.pi)
            log_pref = torch.log(torch.tensor(2.0, device=pred_mean.device)) - torch.log(sigma)
            log_cdf = torch.log(torch.clamp(std_normal.cdf(pred_skew * z), min=1e-12))
            log_pdf = log_pref + log_phi + log_cdf
            loss_drift = -log_pdf.mean()

            mask_logits, mag_pred = jump(trunk, tgt_volume_info)

            # derive binary mask target from sparse jumps (non-zero indicates a jump)
            mask_target = (tgt_sparse.abs() > 1e-8).float()
            # auxiliary BCE loss on mask (keeps supervised signal)
            loss_mask = bce(mask_logits, mask_target)

            # Mixture NLL: point-mass at 0 with prob (1-p) and Gaussian(mu=mag_pred, sigma) with prob p
            p = torch.sigmoid(mask_logits)
            y = tgt_sparse
            mu = mag_pred

            # estimate sigma from non-zero targets if available, else fallback
            nonzero = (y.abs() > 1e-8)
            if nonzero.any():
                sigma_est = y[nonzero].std().detach()
                if isinstance(sigma_est, torch.Tensor) and sigma_est.item() > 1e-6:
                    sigma = sigma_est
                else:
                    sigma = torch.tensor(1.0, device=y.device)
            else:
                sigma = torch.tensor(1.0, device=y.device)

            # compute log Gaussian pdf for all positions
            var = (sigma ** 2)
            log_pdf = -0.5 * ((y - mu) ** 2) / var - 0.5 * math.log(2 * math.pi * var)

            eps = 1e-12
            log_p = torch.log(p + eps)
            log_1mp = torch.log1p(-p + eps)

            is_zero = (y.abs() <= 1e-8)
            # for zero observations: loglik = log( (1-p) + p * N(0;mu,sigma) )
            log_comp_zero = log_p + log_pdf
            # stable log-sum-exp between log_1mp and log_comp_zero
            stacked = torch.stack([log_1mp, log_comp_zero], dim=-1)
            maxv, _ = torch.max(stacked, dim=-1)
            log_mix_zero = maxv + torch.log(torch.exp(stacked[..., 0] - maxv) + torch.exp(stacked[..., 1] - maxv) + eps)
            # for non-zero observations: loglik = log(p) + log_pdf
            log_mix_nonzero = log_p + log_pdf

            # assemble NLL per position
            nll_pos = torch.where(is_zero, -log_mix_zero, -log_mix_nonzero)
            loss_mixture = nll_pos.mean()

            # regularize predicted magnitudes weighted by predicted mask probability
            mag_reg = (p * (mag_pred ** 2)).mean()

            # auxiliary volume alignment loss: encourage p to correlate with future volume info
            vol_align_loss = mse(p, tgt_volume_info)

            # combine losses (weights can be tuned)
            loss = loss_drift + 0.5 * loss_mask + loss_mixture + mag_reg_lambda * mag_reg + vol_align_lambda * vol_align_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_steps += 1

        avg = total_loss / max(1, n_steps)
        train_losses.append(avg)
        print(f'Epoch {ep+1}/{epochs} avg_loss={avg:.6f}')

        # validation visuals: for each validation dataset (>=2024), pick a random window and visualize
        backbone.eval()
        for ds, name in zip(val_datasets, val_names):
            try:
                idx = random.randrange(len(ds))
                dts_v, xs_v, t_v = ds[idx]
                # batchify
                with torch.no_grad():
                    dts_b = dts_v.unsqueeze(0).to(device).float()
                    xs_b = xs_v.unsqueeze(0).to(device).float()
                    emb_v = input_proj(xs_b)
                    trunk_v = backbone(emb_v, dts_b)
                    last_ctx_v = xs_b[:, -1, 0]

                    # move target tensors to device for concatenation
                    if t_v['smooth_path'].shape[0] == 0:
                        continue
                    t_smooth = t_v['smooth_path'].unsqueeze(0).to(device).float()
                    t_volume_info = t_v['volume_information'].unsqueeze(0).to(device).float()
                    t_sparse = t_v['sparse_jumps'].unsqueeze(0).to(device).float()

                    tgt_in_v = torch.cat([last_ctx_v.unsqueeze(1), t_smooth[:, :-1]], dim=1)
                    pm, plv, pls = drift(trunk_v, dts_b, tgt_in_v)
                    pm = pm.squeeze(-1).cpu().numpy()[0]

                    tpath = t_smooth.cpu().numpy()[0]
                    mask_logits_v, mag_v = jump(trunk_v, t_volume_info)
                    mag_pred = mag_v.cpu().numpy()[0]
                    tmag = t_sparse.cpu().numpy()[0][:len(mag_pred)]
                    visualize.plot_predictions(name, f'epoch{ep+1}_{name}.png', tpath, pm, tmag, mag_pred)
            except Exception as e:
                print('Validation viz failed for', name, e)

        visualize.plot_training_curves(train_losses)

        # checkpoint model every epoch and keep best by training loss
        ckpt_dir = os.path.join('outputs', 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'epoch{ep+1}.pt')
        torch.save({
            'epoch': ep + 1,
            'backbone': backbone.state_dict(),
            'input_proj': input_proj.state_dict(),
            'drift': drift.state_dict(),
            'jump': jump.state_dict(),
            'opt': opt.state_dict(),
            'train_losses': train_losses,
        }, ckpt_path)
        if avg < best_loss:
            best_loss = avg
            best_path = os.path.join(ckpt_dir, 'best.pt')
            torch.save({
                'epoch': ep + 1,
                'backbone': backbone.state_dict(),
                'input_proj': input_proj.state_dict(),
                'drift': drift.state_dict(),
                'jump': jump.state_dict(),
                'opt': opt.state_dict(),
                'train_losses': train_losses,
            }, best_path)


if __name__ == '__main__':
    run(epochs=10, use_moe=True, num_experts=3)
