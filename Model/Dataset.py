import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud

class StockDatasetUtils:
    @staticmethod
    def _dejump_target(targets):
        returns = np.diff(targets)
        deviations = np.abs(returns - np.median(returns))
        mad = np.median(deviations) / 0.6745
        threshold = mad * 4.0
        jump_indices = np.where(np.abs(returns) > threshold)[0]
        
        point_jump_returns = np.zeros_like(returns)
        point_jump_returns[jump_indices] = returns[jump_indices]
        continuous_returns = returns - point_jump_returns
        dejumped_close = targets.copy()
        dejumped_close[1:] = np.cumsum(continuous_returns) + targets[0]
        
        return dejumped_close, point_jump_returns

    @staticmethod
    def _standardize_feature(series: np.ndarray, eps=1e-3):
        # Robust standardization: handle NaNs/Infs and very small std
        arr = np.asarray(series, dtype=float)
        # Treat infinities as NaN so nan-ops ignore them
        arr[~np.isfinite(arr)] = np.nan
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        min_std = float(eps)
        if np.isnan(mean):
            mean = 0.0
        if np.isnan(std) or std < min_std:
            std = min_std
        # replace remaining NaNs with the mean before scaling
        filled = np.nan_to_num(arr, nan=mean)
        return (filled - mean) / std, mean, std

    @staticmethod
    def _multi_scale_returns(log_levels, scales=(1, 2, 4, 8, 16)):
        out = np.zeros((*log_levels.shape, len(scales)), dtype=float)
        for i, k in enumerate(scales):
            out[k:, i] = (log_levels[k:] - log_levels[:-k]) / (k ** 0.5)
        return out

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def ffill_nan(x: np.ndarray) -> np.ndarray:
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(len(x)), 0)
        np.maximum.accumulate(idx, out=idx)
        return x[idx]

    @staticmethod
    def fill_series(x: np.ndarray):
        eps = 1e-3
        med = np.nanmedian(x)
        if np.isnan(med):
            med = eps
        x = StockDatasetUtils.ffill_nan(x)
        x = StockDatasetUtils.ffill_nan(x[::-1])[::-1]  # bfill
        x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)
        return x

class StockDataset(tud.Dataset):
    def __init__(self, df: pd.DataFrame, *, stride: int, window_len=192, ctx_len: int=128) -> None:
        self.ctx_len = ctx_len
        self.windows = [
            self._calculate_window_features(df.iloc[offset:offset+window_len])
            for offset in range(0, len(df)-window_len+1, stride)
        ]

    def _calculate_window_features(self, df: pd.DataFrame):
        ctx_len = self.ctx_len
        assert len(df) > ctx_len, "Window length must be greater than context length"
        
        eps = 1e-3
        D = df['D']
        O = StockDatasetUtils.fill_series(df['O'].values)
        H = StockDatasetUtils.fill_series(df['H'].values)
        L = StockDatasetUtils.fill_series(df['L'].values)
        C = StockDatasetUtils.fill_series(df['C'].values)
        V = StockDatasetUtils.fill_series(df['V'].values)
        
        
        dt = (D - D.min()).dt.total_seconds().values
        dt = (dt - dt.min()) / (dt.max() - dt.min() + eps)
        dt = torch.tensor(dt)
        # Ensure finite positive prices before taking logs
        C = np.nan_to_num(C, nan=eps, posinf=1e6, neginf=eps)
        O = np.nan_to_num(O, nan=eps, posinf=1e6, neginf=eps)
        C = np.maximum(C, eps)
        O = np.maximum(O, eps)

        log_C, log_O = np.log(C + eps), np.log(O + eps)
        volume_information = StockDatasetUtils.sigmoid(V / (2 * (np.nanstd(V) + eps)))
        log_C = StockDatasetUtils.ffill_nan(log_C)
        log_C = np.nan_to_num(log_C, nan=0.0)
        log_O = StockDatasetUtils.ffill_nan(log_O)
        log_O = np.nan_to_num(log_O, nan=0.0)

        dejumped_close, point_jump_returns = StockDatasetUtils._dejump_target(log_C)
        
        log_returns = StockDatasetUtils._multi_scale_returns(log_C)
        log_gap = log_O.copy()
        log_gap[1:] = log_O[1:] - log_C[:-1]
        
        candle = (H - L + eps)
        body_ratio = np.abs(C - O) / candle
        upper_wick_ratio = (H - np.maximum(O, C)) / candle
        lower_wick_ratio = (np.minimum(O, C) - L) / candle
        directional_bias = ((2 * C - H - L) / candle).clip(-1, 1)
        directional_bias = np.nan_to_num(directional_bias, nan=0.0, posinf=0.0, neginf=0.0)
        
        directional_proxy = directional_bias * volume_information
        
        # Ensure H and L finite and positive for log-range
        H = np.nan_to_num(H, nan=eps, posinf=1e6, neginf=eps)
        L = np.nan_to_num(L, nan=eps, posinf=1e6, neginf=eps)
        ratio = H / (L + eps)
        ratio = np.nan_to_num(ratio, nan=eps, posinf=1e6, neginf=eps)
        ratio = np.maximum(ratio, eps)
        log_range = np.log(ratio)
        parkinson_vol = (log_range ** 2) / (4 * np.log(2))
        
        dejumped_close_input, dejumped_close_mean, dejumped_close_std = StockDatasetUtils._standardize_feature(
            dejumped_close[:ctx_len]
        )
        dejumped_close_out = (dejumped_close[ctx_len:] - dejumped_close_mean) / (dejumped_close_std + eps)

        # Cyclical time features (weekly, monthly, yearly) computed from datetime index
        # these provide absolute time-scale awareness while retaining periodicity
        # Use smooth sin/cos embeddings; periods: week=7, month~30.4375, year~365.2425
        try:
            weekday = D.dt.weekday.values.astype(float)  # 0-6
            day = D.dt.day.values.astype(float)  # 1-31
            dayofyear = D.dt.dayofyear.values.astype(float)  # 1-366
        except Exception:
            # fallback if D not datetime-like
            n = len(D)
            weekday = np.arange(n) % 7
            day = (np.arange(n) % 30) + 1
            dayofyear = (np.arange(n) % 365) + 1

        weekly_sin = np.sin(2 * np.pi * (weekday / 7.0))
        weekly_cos = np.cos(2 * np.pi * (weekday / 7.0))
        monthly_sin = np.sin(2 * np.pi * ((day - 1) / 30.4375))
        monthly_cos = np.cos(2 * np.pi * ((day - 1) / 30.4375))
        yearly_sin = np.sin(2 * np.pi * ((dayofyear - 1) / 365.2425))
        yearly_cos = np.cos(2 * np.pi * ((dayofyear - 1) / 365.2425))

        
        input_feats = np.column_stack((
            dejumped_close_input,
            point_jump_returns[:ctx_len],
            
            log_returns[:ctx_len], 
            log_gap[:ctx_len], 
            directional_proxy[:ctx_len], 
            
            body_ratio[:ctx_len], 
            upper_wick_ratio[:ctx_len], 
            lower_wick_ratio[:ctx_len], 
            parkinson_vol[:ctx_len],
            # cyclical time features (week, month, year)
            weekly_sin[:ctx_len], weekly_cos[:ctx_len],
            monthly_sin[:ctx_len], monthly_cos[:ctx_len],
            yearly_sin[:ctx_len], yearly_cos[:ctx_len],

            volume_information[:ctx_len]
        ))
        
        input_tensor = torch.tensor(input_feats, dtype=torch.float32)
        # Normalize sparse jumps per-window using max-abs scaling (preserve sign)
        raw_sparse = point_jump_returns[ctx_len-1:]
        max_abs = np.nanmax(np.abs(raw_sparse))
        if not np.isfinite(max_abs) or max_abs < 1e-6:
            denom = 1.0
        else:
            denom = max_abs
        sparse_norm = raw_sparse / denom

        target_tensors = {
            "smooth_path"       : torch.tensor(dejumped_close_out, dtype=torch.float32),
            # normalized to [-1,1] approximately (preserves sign)
            "sparse_jumps"      : torch.tensor(sparse_norm.astype(np.float32)),
            "volume_information": torch.tensor(volume_information[ctx_len:], dtype=torch.float32)
        }
        return dt[:ctx_len], input_tensor, target_tensors

    def __len__(self): return len(self.windows)
    def __getitem__(self, index): return self.windows[index]