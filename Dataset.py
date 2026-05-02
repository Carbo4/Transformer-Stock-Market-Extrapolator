import torch
import pandas as pd
import numpy  as np
from torch.utils.data import Dataset
from typing           import Any, Tuple



OHLCVArrays = Tuple[
    np.ndarray[Any, np.dtype[Any]], 
    np.ndarray[Any, np.dtype[Any]], 
    np.ndarray[Any, np.dtype[Any]], 
    np.ndarray[Any, np.dtype[Any]], 
    np.ndarray[Any, np.dtype[Any]]
]


class DailyOHLCVDataset(Dataset):
    
    """
    PyTorch Dataset for daily OHLCV data with features for encoder–decoder forecasting.

    Args:
        csv_file    : Path to CSV with columns: Date, Open, High, Low, Close, Volume.
        input_len   : Number of encoder days (default 128).
        target_len  : Number of decoder / target days (default 64).
        jump_window : Rolling window size for MAD jump detection (default 60).
        epsilon     : Small value to avoid division by zero / log(0).
    """
    
    def __init__(
        self,
        csv_file    : str,
        input_len   : int   = 128,
        target_len  : int   = 64,
        jump_window : int   = 60,
        epsilon     : float = 1e-8
    ) -> None: 
        self.input_len   = input_len
        self.target_len  = target_len
        self.jump_window = jump_window
        self.epsilon     = epsilon

        # Load and prepare base data
        df = self._load_and_sort(csv_file)
        self.dates = df['Date'].to_numpy(np.float64)
        open, high, low, close, volume = self._extract_ohlcv(df)

        # Normalize OHLC using expanding median of Close
        open_norm, high_norm, low_norm, close_norm = self._normalize_ohlc(open, high, low, close)

        # Log close and returns
        log_close = np.log(close_norm + epsilon)
        ret       = np.diff(log_close, prepend=0.)

        # Detect jumps
        jump_mag, is_jump = self._detect_jumps(ret)

        # Drift and cumulative jump
        cum_jump  = np.cumsum(jump_mag)
        drift_log = log_close - cum_jump
        drift_ret = ret - jump_mag

        # Build all feature groups
        temporal      = self._temporal_features(self.dates)
        informational = self._informational_features(
            volume, open_norm, high_norm, low_norm, close_norm,
            drift_log, drift_ret, cum_jump, epsilon
        )
        rational      = self._rational_features(
            open_norm, high_norm, low_norm, close_norm, volume, epsilon
        )

        # Assemble encoder features, decoder features, and targets
        self.encoder_features = np.column_stack([temporal, informational, rational])
        self.decoder_features = temporal  # only temporal for decoder
        self.targets          = np.column_stack([drift_log, cum_jump])

        # Precompute valid start indices for sliding windows
        self.valid_starts = self._compute_valid_starts(len(self.encoder_features))

    
    # ------
    
    # Data loading helpers
    
    # ------
    
    
    def _load_and_sort(self, csv_file: str) -> pd.DataFrame:
        df = pd.read_csv(
            csv_file, 
            names       = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] ,
            usecols     = range(6)                                           ,
            parse_dates = ['Date']
        )
        df = df.sort_values('Date').reset_index(drop=True)
        return df

    def _extract_ohlcv(self, df: pd.DataFrame) -> OHLCVArrays:
        return (
            df['Open']  .to_numpy(np.float64) ,
            df['High']  .to_numpy(np.float64) ,
            df['Low']   .to_numpy(np.float64) ,
            df['Close'] .to_numpy(np.float64) ,
            df['Volume'].to_numpy(np.float64)  
        )

    
    # ------
    
    # Normalization and jump detection
    
    # ------
    
    
    def _normalize_ohlc(self, open, high, low, close):
        """Divide each OHLC by expanding median of Close (pandas expanding)."""
        med_close = pd.Series(close).expanding().median().values
        med_close = np.maximum(med_close, self.epsilon)  # avoid zero division
        return open / med_close, high / med_close, low / med_close, close / med_close

    def _detect_jumps(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Robust MAD jump detection using pandas rolling.
        Returns:
            jump_mag : array with return value on jump days, 0 otherwise.
            is_jump  : boolean array indicating jump days.
        """
        ret_series = pd.Series(returns)

        # Rolling median (right-aligned, window includes current day)
        rolling_median = ret_series.rolling(window=self.jump_window, min_periods=1).median().values

        # Absolute deviations and rolling MAD
        abs_dev     = np.abs(returns - rolling_median)
        rolling_mad = pd.Series(abs_dev).rolling(window=self.jump_window, min_periods=1).median().values
        assert isinstance(rolling_mad, np.ndarray), f"rolling_mad is of incompatible type {type(rolling_mad)}"
        
        # Threshold (handle MAD=0 to avoid division by zero)
        rr_mad      = np.where(rolling_mad > 0, rolling_mad / 0.6745, 0.0)
        threshold   = 4.0 * rr_mad

        is_jump     = np.abs(returns) > threshold
        jump_mag    = np.where(is_jump, returns, 0.0)
        return jump_mag, is_jump

    
    # ------
    
    # Feature group builders
    
    # ------
    
    
    def _temporal_features(self, dates: np.ndarray) -> np.ndarray:
        """Cyclical encoding of week, month, year using pandas datetime accessors."""
        dt = pd.to_datetime(dates)

        # Day of week (0=Monday,...,6=Sunday)
        dow     = dt.dayofweek.values
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        # Month (1-12)
        month     = dt.month.values
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)

        # Day of year (1-366)
        doy     = dt.dayofyear.values
        doy_sin = np.sin(2 * np.pi * (doy - 1) / 365.25)
        doy_cos = np.cos(2 * np.pi * (doy - 1) / 365.25)

        return np.column_stack([dow_sin, dow_cos, month_sin, month_cos, doy_sin, doy_cos])

    def _informational_features(
        self,
        volume     : np.ndarray,
        open_norm  : np.ndarray,
        high_norm  : np.ndarray,
        low_norm   : np.ndarray,
        close_norm : np.ndarray,
        drift_log  : np.ndarray,
        drift_ret  : np.ndarray,
        cum_jump   : np.ndarray,
        epsilon    : float
    ) -> np.ndarray : 
        """Builds the 6 informational features."""
        # Volume (log)
        log_volume = np.log(volume + 1)

        # Intra-day directional bias
        daily_range = high_norm - low_norm
        intra_bias  = np.zeros_like(close_norm)
        mask        = daily_range > epsilon
        if mask.any(): intra_bias[mask] = (close_norm[mask] - open_norm[mask]) / daily_range[mask]

        # Inter-day directional bias (overnight log return)
        inter_bias     = np.zeros_like(close_norm)
        inter_bias[1:] = np.log(open_norm[1:] / (close_norm[:-1] + epsilon) + epsilon)

        return np.column_stack([
            log_volume,
            drift_log,
            drift_ret,
            cum_jump,
            intra_bias,
            inter_bias
        ])

    def _rational_features(
        self,
        open_norm  : np.ndarray,
        high_norm  : np.ndarray,
        low_norm   : np.ndarray,
        close_norm : np.ndarray,
        volume     : np.ndarray,
        epsilon    : float
    ) -> np.ndarray:
        """Builds the 5 rational features (ratios)."""
        log_price_per_vol = np.log(close_norm + epsilon) - np.log(volume + epsilon) # Log High-Low ratio
        log_hl_ratio      = np.log(high_norm / (low_norm + epsilon) + epsilon)      # Log Price per unit Volume

        # Candle ratios
        daily_range = high_norm - low_norm
        body = np.abs(close_norm - open_norm)

        body_ratio = np.zeros_like(close_norm)
        upper_wick = np.zeros_like(close_norm)
        lower_wick = np.zeros_like(close_norm)

        mask = daily_range > epsilon
        if mask.any():
            body_ratio[mask] = body[mask] / daily_range[mask]
            upper_wick[mask] = (high_norm[mask] - np.maximum(open_norm[mask], close_norm[mask])) / daily_range[mask]
            lower_wick[mask] = (np.minimum(open_norm[mask], close_norm[mask]) - low_norm[mask]) / daily_range[mask]

        return np.column_stack([
            log_price_per_vol ,
            log_hl_ratio      ,
            body_ratio        ,
            upper_wick        ,
            lower_wick
        ])

    
    # ------
    
    # Window management
    
    # ------
    
    
    def _compute_valid_starts(self, total_days: int) -> np.ndarray:
        """Indices where encoder + target fit entirely in the data."""
        return np.arange(total_days - self.input_len - self.target_len + 1)

    def __getitem__(self, idx: int):
        start    = self.valid_starts[idx]
        enc_end  = start + self.input_len
        targ_end = enc_end + self.target_len

        encoder_in = self.encoder_features[start:enc_end]                   # (input_len, 17)
        target     = self.targets[enc_end:targ_end]                         # (target_len, 2)

        decoder_temporal = self.decoder_features[enc_end:targ_end]
        prev_targets     = np.zeros((self.target_len, 2), dtype=np.float64)
        prev_targets[0]  = self.targets[enc_end - 1]                        # shape (2,)
        
        if self.target_len > 1: prev_targets[1:] = target[:-1]

        # Decoder input: concatenate temporal features and previous targets
        decoder_in = np.column_stack([decoder_temporal, prev_targets])      # (target_len, 8)

        return (
            torch.tensor(encoder_in, dtype=torch.float32) ,
            torch.tensor(decoder_in, dtype=torch.float32) ,
            torch.tensor(target, dtype=torch.float32)
        )
        
    def __len__(self) -> int: return len(self.valid_starts)