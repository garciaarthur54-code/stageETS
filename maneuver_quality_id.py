"""
maneuver_quality_id.py
======================
Aircraft system identification pipeline (longitudinal & lateral dynamics):
  FlightData → TrimDetector → MOESPIdentifier → GreyBoxRefinement
  → Metrics → ManeuverQualityAssessor (2311)

VERSION AVEC TRACES DE DEBUG ("[TRACE] ...") — ajoutées pour localiser un
blocage. Cherche la dernière ligne [TRACE] affichée avant que le programme
se fige : c'est l'étape suivante qui bloque.
"""
from __future__ import annotations
import copy
import glob
import os
import time
from dataclasses import dataclass, field, replace as _dataclass_replace
from typing import Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pandas as pd
from scipy.linalg import expm, logm
from scipy.optimize import least_squares
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

try:
    from skimage.restoration import denoise_tv_chambolle as _tv_denoise_sk
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

try:
    from sippy_unipi import system_identification as _sippy_sid
    _HAS_SIPPY = True
except ImportError:
    _HAS_SIPPY = False


try:
    from TrendAnalyzer import TrendAnalyzer as _TrendAnalyzer
    _HAS_TREND_ANALYZER = True
except ImportError:
    _HAS_TREND_ANALYZER = False


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
G_ACC   = 9.80665    # m/s²
KT_MPS  = 0.514444   # knots → m/s
DEG2RAD = np.pi / 180.0

# ─────────────────────────────────────────────────────────────────────────────
# Aircraft column-name presets
# ─────────────────────────────────────────────────────────────────────────────
AIRCRAFT_1: Dict[str, str] = {
    "time":      "time",
    "tas":       "g04_eom_tas_kts_f8",
    "mach":      "g04_eom_mach_f8",
    "alpha":     "g04_eom_alpha_f8",
    "beta":      "g04_eom_beta_f8",
    "p":         "g04_eom_p_deg_f8",
    "q":         "g04_eom_q_deg_f8",
    "r":         "g04_eom_r_deg_f8",
    "theta":     "g04_eom_theta_deg_f8",
    "phi":       "g04_eom_phi_deg_f8",
    "nx":        "g04_eom_nx_f8",
    "ny":        "g04_eom_ny_f8",
    "nz":        "g04_eom_nz_f8",
    "elevator":  "l04_sysi_elv_avg_f8",
    "aileron_l": "l04_sysi_ail_f8(1)",
    "aileron_r": "l04_sysi_ail_f8(2)",
    "rudder":    "l04_sysi_rud_f8(1)",
    "n1p(1)":    "l04_sysi_n1p_f8(1)",
    "n1p(2)" :   "l04_sysi_n1p_f8(2)",
}
AIRCRAFT_2: Dict[str, str] = {
    "time":      "time",
    "tas":       "B640051",
    "mach":      "B640052",
    "alpha":     "g04_eom_alpha_f8_uncal",
    "beta":      "g04_eom_beta_f8",
    "p":         "g04_eom_p_deg_f8",
    "q":         "g04_eom_q_deg_f8",
    "r":         "g04_eom_r_deg_f8",
    "theta":     "g04_eom_theta_deg_f8",
    "phi":       "g04_eom_phi_deg_f8",
    "psi":       "B640036",
    "nx":        "g04_eom_nx_f8",
    "ny":        "g04_eom_ny_f8",
    "nz":        "g04_eom_nz_f8",
    "vns":       "B640065",
    "vew":       "B640066",
    "vz":        "B640067",
    "elevator":  "l04_sysi_elv_avg_f8",
    "aileron_l": "l04_sysi_ail_f8(1)",
    "aileron_r": "l04_sysi_ail_f8(2)",
    "rudder":    "l04_sysi_rud_f8(1)",
    "n1p(1)":    "l04_sysi_n1p_f8(1)",
    "n1p(2)" :   "l04_sysi_n1p_f8(2)",
    "column":    "l04_sysi_colpos_f8",
}
AIRCRAFT_3: Dict[str, str] = {
    "time":      "time",
    "tas":       "g04_eom_tas_kts_f8",
    "mach":      "g04_eom_mach_f8",
    "alpha":     "g04_eom_alpha_f8",
    "beta":      "g04_eom_beta_f8",
    "p":         "g04_eom_p_deg_f8",
    "q":         "g04_eom_q_deg_f8",
    "r":         "g04_eom_r_deg_f8",
    "theta":     "g04_eom_theta_deg_f8",
    "phi":       "g04_eom_phi_deg_f8",
    "nx":        "g04_eom_nx_f8_filt",
    "ny":        "g04_eom_ny_f8_filt",
    "nz":        "g04_eom_nz_f8_filt",
    "elevator":  "l04_sysi_elv_avg_f8",
    "aileron_l": "l04_sysi_ail_f8(1)",
    "aileron_r": "l04_sysi_ail_f8(2)",
    "rudder":    "l04_sysi_rud_f8(1)",
    "n1p(1)":    "l04_sysi_n1p_f8(1)",
    "n1p(2)" :   "l04_sysi_n1p_f8(2)",
}


# ─────────────────────────────────────────────────────────────────────────────
# AircraftConfig
# ─────────────────────────────────────────────────────────────────────────────
class AircraftConfig:
    """Maps canonical variable names to actual DataFrame column names."""

    _PRESETS: Dict[int, Dict] = {1: AIRCRAFT_1, 2: AIRCRAFT_2, 3: AIRCRAFT_3}
    _NZ_CONVENTIONS: Dict[int, int] = {1: -1, 2: 0, 3: 1}

    def __init__(
        self,
        colmap: Dict[str, str],
        g: float = G_ACC,
        kt_mps: float = KT_MPS,
        tas_in_knots: bool = True,
        angles_in_degrees: bool = True,
        plane: int = 2,
        dynamics: str = "longitudinal",
        nz_convention: int = -1,
    ):
        self.colmap = colmap
        self.g = g
        self.kt_mps = kt_mps
        self.tas_in_knots = tas_in_knots
        self.angles_in_degrees = angles_in_degrees
        self.plane = plane
        if dynamics not in ("longitudinal", "lateral", "six_dof"):
            raise ValueError("dynamics doit être 'longitudinal', 'lateral' ou 'six_dof'")
        self.dynamics = dynamics
        self.nz_convention = nz_convention

    @classmethod
    def from_preset(cls, aircraft_id: int, **kwargs) -> "AircraftConfig":
        if aircraft_id not in cls._PRESETS:
            raise ValueError(f"Unknown preset {aircraft_id}. Available: {list(cls._PRESETS)}")
        kwargs.setdefault("plane", aircraft_id)
        kwargs.setdefault("nz_convention", cls._NZ_CONVENTIONS.get(aircraft_id, -1))
        return cls(cls._PRESETS[aircraft_id], **kwargs)

    def col(self, name: str) -> str:
        if name not in self.colmap:
            raise KeyError(f"'{name}' not found in AircraftConfig colmap.")
        return self.colmap[name]

    def get_col(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self.colmap.get(name, default)

    def get_n1_cols(self) -> List[str]:
        return [v for k, v in self.colmap.items() if k.startswith("n1p")]

# ─────────────────────────────────────────────────────────────────────────────
# FlightData
# ─────────────────────────────────────────────────────────────────────────────
class FlightData:
    def __init__(self, df: pd.DataFrame, config: AircraftConfig):
        self.raw_df = df.copy()
        self.config = config
        self._processed = False
        self.df: Optional[pd.DataFrame] = None
        self.time: Optional[np.ndarray] = None
        self.dt: Optional[float] = None
        self.fs: Optional[float] = None
        self._n1_active: bool = False

    @classmethod
    def from_pickle(cls, path: str, config: AircraftConfig) -> "FlightData":
        return cls(pd.read_pickle(path), config)

    @classmethod
    def from_csv(cls, path: str, config: AircraftConfig, **kwargs) -> "FlightData":
        return cls(pd.read_csv(path, **kwargs), config)

    @classmethod
    def from_file(cls, path: str, config: AircraftConfig, **kwargs) -> "FlightData":
        print(f"[TRACE] FlightData.from_file(): ouverture {path}", flush=True)
        path_str = str(path)
        ext = os.path.splitext(path_str)[1].lower()

        if ext in [".pkl", ".pickle"]:
            df = pd.read_pickle(path_str)
        elif ext == ".csv":
            df = pd.read_csv(path_str, **kwargs)
        else:
            raise ValueError(
                f"Extension non supportée: '{ext}'. "
                "Utilise un fichier .pkl, .pickle ou .csv."
            )
        print(f"[TRACE] FlightData.from_file(): fichier lu, {len(df)} lignes", flush=True)
        return cls(df, config)

    @staticmethod
    def _build_time_vector(
        df: pd.DataFrame, default_dt: float
    ) -> Tuple[np.ndarray, float, float]:
        n = len(df)
        if isinstance(df.index, pd.TimedeltaIndex):
            t  = df.index.total_seconds().values
            dt = float(np.median(np.diff(t)))
            return t, dt, 1.0 / dt
        if isinstance(df.index, pd.DatetimeIndex):
            t  = (df.index - df.index[0]).total_seconds().values
            dt = float(np.median(np.diff(t)))
            return t, dt, 1.0 / dt
        for col in ("time", "Time", "TIME", "IRIGB_TIME", "t"):
            if col in df.columns:
                t = pd.to_numeric(df[col], errors="coerce").values
                finite = t[np.isfinite(t)]
                if len(finite) > 1:
                    t  = t - t[np.isfinite(t)][0]
                    dt = float(np.median(np.diff(finite)))
                    return t, dt, 1.0 / dt
        t  = np.arange(n, dtype=float) * default_dt
        return t, default_dt, 1.0 / default_dt

    @staticmethod
    def _tv_denoise(signal: np.ndarray, weight: float) -> np.ndarray:
        if _HAS_SKIMAGE:
            return _tv_denoise_sk(signal.astype(float), weight=weight)
        from scipy.ndimage import uniform_filter1d
        width = max(3, int(weight * len(signal) * 0.1))
        return uniform_filter1d(signal.astype(float), size=width)

    @staticmethod
    def _lowpass(y: np.ndarray, dt: float, fc_hz: float, order: int = 4) -> np.ndarray:
        from scipy.signal import butter, filtfilt

        y = np.asarray(y, dtype=float)
        fin = np.isfinite(y)
        if fin.sum() < 3 * order:
            return np.full_like(y, np.nan)
        nyq = 0.5 / dt
        wn  = min(fc_hz / nyq, 0.99)
        b, a = butter(order, wn, btype="low")
        y_filled = np.where(fin, y, np.nanmean(y[fin]))
        y_filt = filtfilt(b, a, y_filled)
        y_filt[~fin] = np.nan
        return y_filt

    @staticmethod
    def _hampel_filter(y: np.ndarray, window_n: int, n_sigma: float = 4.0) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        fin = np.isfinite(y)
        s = pd.Series(y)
        med = s.rolling(window_n, center=True, min_periods=1).median()
        resid = (s - med).abs()
        mad = resid.rolling(window_n, center=True, min_periods=1).median()
        is_outlier = (resid > n_sigma * 1.4826 * mad).to_numpy()
        out = np.where(is_outlier, med.to_numpy(), y)
        out[~fin] = np.nan
        return out

    def preprocess(
        self,
        tv_weight: float = 20.0,
        default_dt: float = 0.0625,
        n1p_variation_threshold: float = 0.02,
        use_accel_lowpass: bool = False,
        accel_lowpass_fc_hz: float = 0.5,
        verbose: bool = False,
    ) -> "FlightData":
        print("[TRACE] preprocess(): début", flush=True)
        cfg = self.config
        df = self.raw_df.copy()

        # 1 – time vector
        self.time, self.dt, self.fs = FlightData._build_time_vector(df, default_dt)
        df["_time"] = self.time
        print(f"[TRACE] preprocess(): time vector OK, N={len(df)}, dt={self.dt:.4f}", flush=True)

        # 2 – TV derivatives for KDE scoring
        for key in ("elevator", "aileron_l", "aileron_r", "rudder"):
            col = cfg.get_col(key)
            if col and col in df.columns:
                tv_smooth = FlightData._tv_denoise(df[col].values.astype(float), tv_weight)
                df[f"{col}_diff_tv"] = 1000.0 * np.gradient(tv_smooth, self.dt)
        print("[TRACE] preprocess(): TV-denoise commandes OK", flush=True)

        if cfg.dynamics == "lateral":
            ail_l = cfg.col("aileron_l")
            ail_r = cfg.col("aileron_r")
            rud   = cfg.col("rudder")

            df["_ail_diff"] = 0.5 * (df[ail_l].values - df[ail_r].values)
            df["_ail_diff_rate"] = np.gradient(df["_ail_diff"].values, self.dt)
            df["_rudder_rate"] = np.gradient(df[rud].values, self.dt)
        else:
            elv_col = cfg.col("elevator")
            df["_elv_rate"] = np.gradient(df[elv_col].values, self.dt)

        # 3 – Body-axis velocity reconstruction
        tas_col   = cfg.col("tas")
        alpha_col = cfg.col("alpha")
        beta_col  = cfg.get_col("beta")

        tas = df[tas_col].values
        if cfg.tas_in_knots:
            tas = tas * cfg.kt_mps

        alpha_raw = df[alpha_col].values
        alpha_rad = alpha_raw * DEG2RAD if cfg.angles_in_degrees else alpha_raw

        if beta_col and beta_col in df.columns:
            beta_raw = df[beta_col].values
            beta_rad = beta_raw * DEG2RAD if cfg.angles_in_degrees else beta_raw
        else:
            beta_rad = np.zeros(len(df))

        df["_u"] = tas * np.cos(alpha_rad) * np.cos(beta_rad)
        df["_w"] = tas * np.sin(alpha_rad) * np.cos(beta_rad)
        df["_v"] = tas * np.sin(beta_rad)
        df["_alpha_rad"] = alpha_rad
        df["_beta_rad"]  = beta_rad
        print("[TRACE] preprocess(): reconstruction vitesses corps OK", flush=True)

        # 4 – Angular states to radians
        for key in ("p", "q", "r", "theta", "phi"):
            col = cfg.get_col(key)
            if col and col in df.columns and cfg.angles_in_degrees:
                df[col] = df[col].values * DEG2RAD

        # 5 – Normalize N1%
        n1_raw_cols = [c for c in cfg.get_n1_cols() if c in df.columns]
        if n1_raw_cols:
            n1_avg = np.column_stack([df[c].values for c in n1_raw_cols]).mean(axis=1)
            df["_n1p"] = n1_avg / 100.0
        else:
            df["_n1p"] = np.zeros(len(df))

        # 6 – Body accelerations + passe-bas
        self.use_accel_lowpass  = use_accel_lowpass
        self.accel_lowpass_fc_hz = accel_lowpass_fc_hz
        for key, out in (("nx", "_ax"), ("ny", "_ay"), ("nz", "_az")):
            col = cfg.get_col(key)
            if col and col in df.columns:
                df[out] = df[col].values * cfg.g
                df[f"{out}_lp"] = FlightData._lowpass(df[out].values, self.dt, accel_lowpass_fc_hz)
        print("[TRACE] preprocess(): accélérations corps + passe-bas OK", flush=True)

        # 6bis – Même passe-bas pour X, U
        recon_cols = {"_u", "_v", "_w", "_ail_diff", "_n1p"}
        for key in ("p", "q", "r", "phi", "theta", "elevator", "rudder"):
            c = cfg.get_col(key)
            if c:
                recon_cols.add(c)
        for c in recon_cols:
            if c in df.columns:
                df[f"{c}_lp"] = FlightData._lowpass(df[c].values.astype(float), self.dt, accel_lowpass_fc_hz)
        print("[TRACE] preprocess(): passe-bas X/U OK", flush=True)

        # 7 – Active N1 detection
        self._n1_active = False
        if n1_raw_cols:
            self._n1_active = (df["_n1p"].max() - df["_n1p"].min()) > n1p_variation_threshold

        self.df = df
        self._processed = True

        if verbose:
            print(
                f"[FlightData] N={len(df)}, dt={self.dt:.4f} s, "
                f"fs={self.fs:.2f} Hz, N1 active={self._n1_active}"
            )
        print("[TRACE] preprocess(): FIN", flush=True)
        return self

    def _series(self, col: str) -> np.ndarray:
        if getattr(self, "use_accel_lowpass", False):
            lp_col = f"{col}_lp"
            if lp_col in self.df.columns:
                return self.df[lp_col].values
        return self.df[col].values

    def _require(self):
        if not self._processed:
            raise RuntimeError("Call .preprocess() before accessing data arrays.")

    @property
    def X(self) -> np.ndarray:
        self._require()
        cfg = self.config
        if cfg.dynamics == "lateral":
            return np.column_stack([
                self._series("_v"),
                self._series(cfg.col("p")),
                self._series(cfg.col("r")),
                self._series(cfg.col("phi")),
            ])
        if cfg.dynamics == "six_dof":
            return np.column_stack([
                self._series("_u"),
                self._series("_v"),
                self._series("_w"),
                self._series(cfg.col("p")),
                self._series(cfg.col("q")),
                self._series(cfg.col("r")),
                self._series(cfg.col("phi")),
                self._series(cfg.col("theta")),
            ])
        return np.column_stack([
            self._series("_u"),
            self._series("_w"),
            self._series(cfg.col("q")),
            self._series(cfg.col("theta")),
        ])

    @property
    def U(self) -> np.ndarray:
        self._require()
        cfg = self.config
        if cfg.dynamics == "lateral":
            return np.column_stack([
                self._series("_ail_diff"),
                self._series(cfg.col("rudder")),
            ])
        if cfg.dynamics == "six_dof":
            n1  = self._series("_n1p")      if "_n1p"      in self.df.columns else np.zeros(len(self.df))
            ail = self._series("_ail_diff") if "_ail_diff" in self.df.columns else np.zeros(len(self.df))
            return np.column_stack([
                self._series(cfg.col("elevator")),
                n1,
                ail,
                self._series(cfg.col("rudder")),
            ])
        elv_col = cfg.col("elevator")
        U = self._series(elv_col).reshape(-1, 1)
        if self._n1_active:
            U = np.column_stack([U, self._series("_n1p")])
        return U

    @property
    def Acc(self) -> np.ndarray:
        self._require()
        if self.config.dynamics == "lateral":
            return self._series("_ay").reshape(-1, 1)
        if self.config.dynamics == "six_dof":
            return np.column_stack([
                self._series("_ax"),
                self._series("_ay"),
                self._series("_az"),
            ])
        return np.column_stack([self._series("_ax"), self._series("_az")])

    @property
    def nu(self) -> int:
        return self.U.shape[1]

    def center(
        self,
        X_mean: np.ndarray,
        U_mean: np.ndarray,
        Acc_mean: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.X - X_mean, self.U - U_mean, self.Acc - Acc_mean

    def clean_mask(self, Xc: np.ndarray, Uc: np.ndarray) -> np.ndarray:
        return np.all(np.isfinite(Xc), axis=1) & np.all(np.isfinite(Uc), axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# TrimResult
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrimResult:
    i0: int
    i1: int
    i0_avg: int
    i1_avg: int
    X_mean: np.ndarray
    U_mean: np.ndarray
    Acc_mean: np.ndarray
    alpha_mean: float = 0.0
    mach_mean: float  = 0.0
    scores_df: Optional[pd.DataFrame] = None
    score_col: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# TrimDetector
# ─────────────────────────────────────────────────────────────────────────────
class TrimDetector:
    """Detects the pre-maneuver trim segment using a KDE-based classifier (hyb)."""

    _THRESHOLDS = [0.95, 0.90, 0.85, 0.80]

    def __init__(
        self,
        hyb=None,
        trim_avg_from_end_s: Optional[float] = 10.0,
    ):
        self.hyb = hyb
        self.trim_avg_from_end_s = trim_avg_from_end_s

    @classmethod
    def load_hyb(cls, best_models_dir, which="total", version_tag="V8"):
        which = which.lower().strip()
        if which not in {"aileron", "rudder", "elevator", "total"}:
            raise ValueError("which doit être: 'aileron', 'rudder', 'elevator', 'total'")
        pattern = os.path.join(best_models_dir, f"best_model_{which}_{version_tag}*cloudpkl*")
        hits = sorted(glob.glob(pattern))
        if not hits:
            pattern2 = os.path.join(best_models_dir, f"best_model_{which}_{version_tag}*.pkl")
            hits = sorted(glob.glob(pattern2))
        if not hits:
            raise FileNotFoundError(
                f"Modèle introuvable.\nPatterns testés:\n- {pattern}\n- {pattern2}"
            )
        model_path = hits[0]
        with open(model_path, "rb") as f:
            hyb = cloudpickle.load(f)
        print(f"✅ Modèle chargé: {model_path}")
        return hyb

    def _auto_downsample_for_trim(
        self,
        data: "FlightData",
        target_duration_s: float = 60.0,
        max_ds: int = 20,
    ):
        N = len(data.df)
        dt = float(data.dt)
        duration_s = N * dt

        if duration_s <= target_duration_s:
            return data, 1

        ds = int(np.ceil(duration_s / target_duration_s))
        ds = max(1, min(ds, max_ds))

        data_ds = copy.copy(data)

        data_ds.df = data.df.iloc[::ds].reset_index(drop=True)
        data_ds.raw_df = data.raw_df.iloc[::ds].reset_index(drop=True) if hasattr(data, "raw_df") else data_ds.df
        data_ds.dt = data.dt * ds

        return data_ds, ds

    def detect(
        self,
        data: FlightData,
        plane: int = 2,
        refine_with_elevator: bool = True,
        auto_downsample: bool = True,
        target_trim_duration_s: float = 20.0,
    ) -> TrimResult:

        if auto_downsample:
            data_for_trim, ds_trim = self._auto_downsample_for_trim(
                data,
                target_duration_s=target_trim_duration_s,
            )
        else:
            data_for_trim, ds_trim = data, 1

        score_priority = self._score_priority_for_plane(plane)

        last_error = None

        for threshold in self._THRESHOLDS:
            print(f"[TRACE] detect(): essai threshold={threshold}", flush=True)
            try:
                trim_ds = self._detect_explicit(
                    data_for_trim,
                    threshold=threshold,
                    score_col_priority=score_priority,
                    avg_duration_s=self.trim_avg_from_end_s or 10.0,
                    is_last_try=(threshold == self._THRESHOLDS[-1]),
                    plane=plane,
                    refine_with_elevator=refine_with_elevator,
                )
                print(f"[TRACE] detect(): threshold={threshold} OK", flush=True)

                if ds_trim == 1:
                    return trim_ds

                return self._rescale_trim_result_to_original(
                    trim_ds,
                    data_original=data,
                    ds_trim=ds_trim,
                )

            except RuntimeError as exc:
                print(f"[TRACE] detect(): threshold={threshold} ECHEC -> {exc}", flush=True)
                last_error = exc
                continue

        raise RuntimeError(f"Trim introuvable à tous les seuils. Dernière erreur: {last_error}")

    def _detect_explicit(
        self,
        data: "FlightData",
        threshold: float,
        score_col_priority: Tuple[Union[str, Tuple[str, ...]], ...],
        avg_duration_s: float,
        is_last_try: bool,
        plane: int,
        min_duration_s: float = 3.0,
        refine_with_elevator: bool = True,
        maneuver_name: str = "",
    ) -> "TrimResult":
        print(f"[TRACE] _detect_explicit(): entrée, threshold={threshold}", flush=True)
        data_orig = data
        data, ds_trim = self._auto_downsample_for_trim(data)
        print(f"[TRACE] _detect_explicit(): après downsample, ds_trim={ds_trim}, N={len(data.df)}", flush=True)

        df  = data.df
        dt  = data.dt
        cfg = data.config

        df_hyb = self._prepare_df_for_hyb(df, cfg)
        print("[TRACE] _detect_explicit(): appel _find_initial_trim_segment_from_kde...", flush=True)
        try:
            seg, scores_df, score_col = self._find_initial_trim_segment_from_kde(
                df_hyb, self.hyb,
                score_col_priority=score_col_priority,
                threshold=threshold,
                min_duration_s=min_duration_s,
                dt=dt,
            )
        except (ValueError, Exception) as exc:
            raise RuntimeError(str(exc)) from exc
        print(f"[TRACE] _detect_explicit(): _find_initial_trim_segment_from_kde terminé, seg={seg}", flush=True)

        if seg is None:
            raise RuntimeError(
                f"Trim introuvable (score={score_col_priority}, thr={threshold:.3f})"
            )

        i0, i1 = seg

        if refine_with_elevator:
            print("[TRACE] _detect_explicit(): refine_with_elevator...", flush=True)
            refine_col = TrimDetector._refine_column_for_maneuver(cfg, maneuver_name)
            refine_n_sigma = TrimDetector._refine_n_sigma_for_column(cfg, refine_col)
            i1 = TrimDetector._refine_trim_end_control(
                df, i0, i1, dt, refine_col, n_sigma=refine_n_sigma
            )
            print("[TRACE] _detect_explicit(): refine terminé", flush=True)

        trim_len = i1 - i0
        trim_len_s = trim_len * dt
        avg_duration_s = min(avg_duration_s, trim_len_s)
        n_last   = max(20, int(np.ceil(avg_duration_s / max(dt, 1e-12))))

        if trim_len >= n_last:
            if is_last_try:
                i0_avg = i0;                i1_avg = i0 + n_last
            else:
                i0_avg = i1 - n_last;       i1_avg = i1
        else:
            i0_avg, i1_avg = i0, i1

        sl_avg = slice(i0_avg, i1_avg)
        X, U, Acc = data.X, data.U, data.Acc

        alpha_col = cfg.col("alpha")
        mach_col  = cfg.get_col("mach")

        alpha_mean = float(
            np.deg2rad(data.raw_df[alpha_col].values[sl_avg].mean())
            if cfg.angles_in_degrees
            else df["_alpha_rad"].values[sl_avg].mean()
        )
        mach_mean = (
            float(df[mach_col].values[sl_avg].mean())
            if mach_col and mach_col in df.columns else 0.0
        )

        trim_ds = TrimResult(
            i0=i0, i1=i1,
            i0_avg=i0_avg, i1_avg=i1_avg,
            X_mean=X[sl_avg].mean(axis=0),
            U_mean=U[sl_avg].mean(axis=0),
            Acc_mean=Acc[sl_avg].mean(axis=0),
            alpha_mean=alpha_mean,
            mach_mean=mach_mean,
            scores_df=scores_df,
            score_col=score_col,
        )

        print("[TRACE] _detect_explicit(): sortie OK", flush=True)
        if ds_trim == 1:
            return trim_ds
        return self._rescale_trim_result_to_original(trim_ds, data_orig, ds_trim)

    @staticmethod
    def _resample_scores_df(scores_df: Optional[pd.DataFrame], target_len: int) -> Optional[pd.DataFrame]:
        if scores_df is None or len(scores_df) == target_len:
            return scores_df
        src_idx = np.arange(len(scores_df))
        dst_idx = np.linspace(0, len(scores_df) - 1, target_len)
        resampled = {
            col: np.interp(dst_idx, src_idx, pd.to_numeric(scores_df[col], errors="coerce").to_numpy(float))
            for col in scores_df.columns
        }
        return pd.DataFrame(resampled)

    def _rescale_trim_result_to_original(
        self,
        trim_ds: "TrimResult",
        data_original: "FlightData",
        ds_trim: int,
    ) -> "TrimResult":
        N = len(data_original.df)

        i0 = min(max(trim_ds.i0 * ds_trim, 0), N - 1)
        i1 = min(max(trim_ds.i1 * ds_trim, i0 + 1), N)

        i0_avg = min(max(trim_ds.i0_avg * ds_trim, i0), N - 1)
        i1_avg = min(max(trim_ds.i1_avg * ds_trim, i0_avg + 1), N)

        sl_avg = slice(i0_avg, i1_avg)

        cfg = data_original.config
        df = data_original.df

        alpha_col = cfg.col("alpha")
        mach_col = cfg.get_col("mach")

        alpha_mean = float(
            np.deg2rad(data_original.raw_df[alpha_col].values[sl_avg].mean())
            if cfg.angles_in_degrees
            else df["_alpha_rad"].values[sl_avg].mean()
        )

        mach_mean = (
            float(df[mach_col].values[sl_avg].mean())
            if mach_col and mach_col in df.columns else 0.0
        )

        return TrimResult(
            i0=i0,
            i1=i1,
            i0_avg=i0_avg,
            i1_avg=i1_avg,
            X_mean=data_original.X[sl_avg].mean(axis=0),
            U_mean=data_original.U[sl_avg].mean(axis=0),
            Acc_mean=data_original.Acc[sl_avg].mean(axis=0),
            alpha_mean=alpha_mean,
            mach_mean=mach_mean,
            scores_df=self._resample_scores_df(trim_ds.scores_df, N),
            score_col=trim_ds.score_col,
        )

    def _score_priority_for_plane(self, plane: int, dynamics: str = "longitudinal",
                                   maneuver_type: str = "2311",
                                   maneuver_name: str = "") -> Tuple[Union[str, Tuple[str, ...]], ...]:
        if dynamics == "lateral":
            return (
                ("diff_aileron1_1d", "diff_aileron2_1d", "diff_rudder_1d"),
                "_COMBINED_",
                "combined_wmean",
                "combined",
                "accel_3d",
                "score",
            )

        if dynamics == "six_dof":
            return ("accel_3d", "diff_elevator_1d", "_COMBINED_", "score")

        if maneuver_type in ("column_sweep", "frequency_sweep"):
            return ("accel_3d", "_COMBINED_", "combined_wmean", "score")

        if plane == 2:
            if maneuver_type == "phugoid":
                return ("_COMBINED_","diff_elevator_1d", "accel_3d",  "score")
            return ("diff_elevator_1d", "_COMBINED_", "accel_3d")

        if plane == 3:
            return ("diff_elevator_1d", "_COMBINED_", "accel_3d")

        return ("score", "accel_3d")

    @staticmethod
    def _first_true_segment(mask, min_len=20):
        mask = np.asarray(mask, dtype=bool)
        n = len(mask)
        if n == 0:
            return None

        padded = np.concatenate(([False], mask, [False]))
        edges  = np.diff(padded.astype(np.int8))
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]

        long_enough = np.where((ends - starts) >= min_len)[0]
        if len(long_enough) == 0:
            return None
        idx = long_enough[0]
        return int(starts[idx]), int(ends[idx])

    @staticmethod
    def _find_initial_trim_segment_from_kde(
        df,
        hyb,
        score_col_priority=("_COMBINED_", "combined_wmean", "combined", "score", "accel_3d", "lp01_smooth", "lp01_agg"),
        threshold=0.95,
        min_duration_s=2.0,
        dt=0.01,
    ):
        if hyb is None:
            return None, None, None

        print(f"[TRACE] >>> appel hyb.score(df)  N={len(df)} ...", flush=True)
        scores = hyb.score(df)
        print("[TRACE] <<< hyb.score(df) terminé", flush=True)

        per_map_df = None
        combined_df = None
        scores_df = None
        score_col_used = None

        if isinstance(scores, (tuple, list)) and len(scores) == 2:
            per_map, combined = scores

            if isinstance(per_map, dict):
                per_map_series = {}
                for map_name, bundle in per_map.items():
                    if isinstance(bundle, dict):
                        if "lp01_smooth" in bundle:
                            s = bundle["lp01_smooth"]
                        elif "lp01_agg" in bundle:
                            s = bundle["lp01_agg"]
                        elif "lp01" in bundle:
                            s = bundle["lp01"]
                        elif "score" in bundle:
                            s = bundle["score"]
                        else:
                            continue
                        if isinstance(s, pd.Series):
                            per_map_series[map_name] = s.rename(map_name)
                        else:
                            arr = np.asarray(s).ravel()
                            if len(arr) == len(df):
                                per_map_series[map_name] = pd.Series(arr, index=df.index, name=map_name)
                if len(per_map_series) > 0:
                    per_map_df = pd.DataFrame(per_map_series)
            elif isinstance(per_map, pd.DataFrame):
                per_map_df = per_map.copy()
            elif isinstance(per_map, pd.Series):
                per_map_df = per_map.to_frame("score")

            if isinstance(combined, pd.DataFrame):
                combined_df = combined.copy()
            elif isinstance(combined, pd.Series):
                cname = combined.name if combined.name is not None else "_COMBINED_"
                combined_df = combined.to_frame(cname)
            else:
                arr = np.asarray(combined).ravel()
                if len(arr) == len(df):
                    combined_df = pd.DataFrame({"_COMBINED_": arr}, index=df.index)

            if combined_df is not None and combined_df.shape[1] > 0:
                scores_df = combined_df.copy()
                if "_COMBINED_" not in scores_df.columns:
                    last_col = scores_df.columns[-1]
                    scores_df["_COMBINED_"] = pd.to_numeric(scores_df[last_col], errors="coerce")
            elif per_map_df is not None:
                scores_df = per_map_df.copy()

        elif isinstance(scores, pd.DataFrame):
            scores_df = scores.copy()

        elif isinstance(scores, pd.Series):
            cname = scores.name if scores.name is not None else "score"
            scores_df = scores.to_frame(cname)

        elif isinstance(scores, dict):
            if "combined" in scores:
                combined = scores["combined"]
                if isinstance(combined, pd.DataFrame):
                    scores_df = combined.copy()
                    if "_COMBINED_" not in scores_df.columns and scores_df.shape[1] > 0:
                        last_col = scores_df.columns[-1]
                        scores_df["_COMBINED_"] = pd.to_numeric(scores_df[last_col], errors="coerce")
                elif isinstance(combined, pd.Series):
                    cname = combined.name if combined.name is not None else "_COMBINED_"
                    scores_df = combined.to_frame(cname)
                    if "_COMBINED_" not in scores_df.columns:
                        scores_df["_COMBINED_"] = pd.to_numeric(scores_df.iloc[:, 0], errors="coerce")
            else:
                tmp = {}
                for k, v in scores.items():
                    if isinstance(v, pd.Series):
                        tmp[k] = v.rename(k)
                    elif isinstance(v, pd.DataFrame) and v.shape[1] >= 1:
                        tmp[k] = v.iloc[:, 0].rename(k)
                    else:
                        arr = np.asarray(v).ravel()
                        if len(arr) == len(df):
                            tmp[k] = pd.Series(arr, index=df.index, name=k)
                if len(tmp) > 0:
                    scores_df = pd.DataFrame(tmp)

        else:
            arr = np.asarray(scores).ravel()
            if len(arr) != len(df):
                raise ValueError(
                    f"hyb.score(df) doit renvoyer une structure compatible de longueur {len(df)}. "
                    f"Reçu shape={getattr(np.asarray(scores), 'shape', None)}"
                )
            scores_df = pd.DataFrame({"score": arr}, index=df.index)

        if scores_df is None or len(scores_df) == 0:
            raise ValueError("Aucun score exploitable trouvé dans hyb.score(df).")

        num_cols = [c for c in scores_df.columns if pd.api.types.is_numeric_dtype(scores_df[c])]
        if len(num_cols) == 0:
            scores_df = scores_df.apply(pd.to_numeric, errors="coerce")
            num_cols = [c for c in scores_df.columns if pd.api.types.is_numeric_dtype(scores_df[c])]
        if len(num_cols) == 0:
            raise ValueError(f"Aucune colonne numérique trouvée dans scores_df: {scores_df.columns.tolist()}")
        scores_df = scores_df[num_cols].copy()

        for c in score_col_priority:
            if isinstance(c, (tuple, list)):
                present = [col for col in c if col in scores_df.columns]
                if present:
                    combo_name = "min(" + ",".join(present) + ")"
                    scores_df[combo_name] = scores_df[present].min(axis=1)
                    score_col_used = combo_name
                    break
            elif c in scores_df.columns:
                score_col_used = c
                break
        if score_col_used is None and "_COMBINED_" in scores_df.columns:
            score_col_used = "_COMBINED_"
        if score_col_used is None:
            score_col_used = scores_df.columns[0]

        score_vec = pd.to_numeric(scores_df[score_col_used], errors="coerce").to_numpy(dtype=float)
        trim_mask = np.isfinite(score_vec) & (score_vec >= float(threshold))
        min_len = max(3, int(np.ceil(min_duration_s / max(dt, 1e-12))))
        seg = TrimDetector._first_true_segment(trim_mask, min_len=min_len)

        return seg, scores_df, score_col_used

    @staticmethod
    def _refine_column_for_maneuver(cfg, maneuver_name: str) -> str:
        """Choisit la colonne de commande à utiliser pour raffiner le segment
        de trim, en fonction du nom de la manœuvre.

        Mapping strict :
          - "wheel"  -> ailerons uniquement (aileron_l)
          - "pedal"  -> rudder uniquement
          - "column" -> elevator uniquement
        Ces trois mots-clés sont prioritaires et exclusifs les uns des
        autres. Les synonymes usuels (elevator/rudder/aileron en toutes
        lettres) restent reconnus en complément pour les noms de manœuvre
        qui ne contiennent pas wheel/pedal/column.
        """
        name = (maneuver_name or "").lower()
        if "wheel" in name:
            return cfg.col("aileron_l")
        if "pedal" in name:
            return cfg.col("rudder")
        if "column" in name:
            return cfg.col("elevator")
        if any(k in name for k in ("elevator", "elevateur", "élévateur")):
            return cfg.col("elevator")
        if "rudder" in name:
            return cfg.col("rudder")
        if "aileron" in name:
            return cfg.col("aileron_l")
        return cfg.col("aileron_l") if cfg.dynamics == "lateral" else cfg.col("elevator")

    @staticmethod
    def _refine_n_sigma_for_column(cfg, refine_col: str) -> float:
        return 10.0 if refine_col == cfg.get_col("elevator") else 6.0

    @staticmethod
    def _refine_trim_end_control(
        df: pd.DataFrame,
        i0: int,
        i1: int,
        dt: float,
        control_col: str,
        n_sigma: float = 6.0,
        min_sustain_s: float = 0.5,
        baseline_s: float = 3.0,
        extend_s: float = 5.0,
        lookback_s: float = 0.0,
        baseline_i0: Optional[int] = None,
        baseline_i1: Optional[int] = None,
    ) -> int:
        if control_col not in df.columns:
            return i1
        full = df[control_col].values
        i0_search = max(0, i0 - int(round(lookback_s / dt))) if lookback_s > 0 else i0
        i1_search = min(len(df), i1 + int(round(extend_s / dt)))
        if baseline_i0 is not None and baseline_i1 is not None:
            base = full[baseline_i0:baseline_i1]
        else:
            elv_fwd = full[i0:i1_search]
            n_base  = min(len(elv_fwd), max(3, int(np.ceil(baseline_s / dt))))
            base    = elv_fwd[:n_base]
        if len(base) == 0:
            return i1
        mu    = base.mean()
        sigma = base.std()
        if sigma < 1e-9:
            return i1
        elv      = full[i0_search:i1_search]
        outside  = np.abs(elv - mu) > n_sigma * sigma
        min_sust = max(1, int(np.ceil(min_sustain_s / dt)))
        if len(outside) <= min_sust:
            return i1
        window_sum = np.convolve(outside.astype(int), np.ones(min_sust, dtype=int), mode="valid")[:-1]
        hit = np.where(window_sum == min_sust)[0]
        return i0_search + int(hit[0]) if len(hit) else i1

    @staticmethod
    def _prepare_df_for_hyb(df: pd.DataFrame, config) -> pd.DataFrame:
        _G_SI = 9.80665
        df = df.copy()

        theta_col = config.get_col("theta")
        phi_col   = config.get_col("phi")
        theta_rad = phi_rad = None

        if theta_col and theta_col in df.columns:
            theta_rad = df[theta_col].values
            df["sin_theta"] = np.sin(theta_rad)
            df["cos_theta"] = np.cos(theta_rad)

        if phi_col and phi_col in df.columns:
            phi_rad = df[phi_col].values
            df["sin_phi"] = np.sin(phi_rad)
            df["cos_phi"] = np.cos(phi_rad)

        if theta_rad is not None and phi_rad is not None:
            df["cos_theta_sin_phi"] = np.cos(theta_rad) * np.sin(phi_rad)
            df["cos_theta_cos_phi"] = np.cos(theta_rad) * np.cos(phi_rad)
            df["sin_theta_sin_phi"] = np.sin(theta_rad) * np.sin(phi_rad)
            df["sin_theta_cos_phi"] = np.sin(theta_rad) * np.cos(phi_rad)

        p_col  = config.get_col("p")
        q_col  = config.get_col("q")
        r_col  = config.get_col("r")
        nx_col = config.get_col("nx")
        ny_col = config.get_col("ny")
        nz_col = config.get_col("nz")
        tas_col = config.get_col("tas")
        plane   = getattr(config, "plane", 2)

        body_cols_present = all(
            c and c in df.columns
            for c in [p_col, q_col, r_col, nx_col, ny_col, nz_col]
        )

        if body_cols_present and theta_rad is not None and phi_rad is not None \
                and "_u" in df.columns and "_w" in df.columns:
            p  = df[p_col].values
            q  = df[q_col].values
            r  = df[r_col].values
            nx = df[nx_col].values
            ny = df[ny_col].values
            nz = df[nz_col].values
            U_g = df["_u"].values
            W_g = df["_w"].values
            beta_rad = df["_beta_rad"].values if "_beta_rad" in df.columns else np.zeros(len(df))
            if tas_col and tas_col in df.columns:
                V_g = df[tas_col].values * np.sin(beta_rad)
            else:
                V_g = np.zeros(len(df))

            df["Body x-axis acceleration Up"] = (
                r * V_g - q * W_g - _G_SI * np.sin(theta_rad) + nx * _G_SI
            )
            df["Body y-axis acceleration Vp"] = (
                p * W_g - r * U_g
                + _G_SI * np.cos(theta_rad) * np.sin(phi_rad)
                + ny * _G_SI
            )
            nz_convention = getattr(config, "nz_convention", -1)
            Wp = (
                q * U_g - p * V_g
                + _G_SI * np.cos(theta_rad) * np.cos(phi_rad)
                + nz * _G_SI
            )
            Wp -= (1 + nz_convention) * _G_SI
            df["Body z-axis acceleration Wp"] = Wp

        return df

# ─────────────────────────────────────────────────────────────────────────────
# StateSpaceModel
# ─────────────────────────────────────────────────────────────────────────────
class StateSpaceModel:
    def __init__(
        self,
        Ac: np.ndarray,
        Bc: np.ndarray,
        nu: int = 1,
        u0: float = 0.0,
        theta0: float = 0.0,
        w0: float = 0.0,
        g: float = G_ACC,
    ):
        self.Ac     = np.asarray(Ac, dtype=float)
        self.Bc     = np.asarray(Bc, dtype=float)
        self.nx     = self.Ac.shape[0]
        self.nu     = nu
        self.u0     = u0
        self.theta0 = theta0
        self.w0     = w0
        self.g      = g
        self.F:  Optional[np.ndarray] = None
        self.G:  Optional[np.ndarray] = None
        self._dt: Optional[float]     = None

    def discretize(self, dt: float) -> "StateSpaceModel":
        self.F, self.G = StateSpaceModel.zoh(self.Ac, self.Bc, dt)
        self._dt = dt
        return self

    def simulate(
        self, x0: np.ndarray, U: np.ndarray, dt: Optional[float] = None
    ) -> np.ndarray:
        if dt is not None and dt != self._dt:
            self.discretize(dt)
        if self.F is None:
            raise RuntimeError("Call .discretize(dt) before .simulate().")
        N = len(U)
        X = np.zeros((N + 1, self.nx))
        X[0] = x0
        for k in range(N):
            X[k + 1] = self.F @ np.nan_to_num(X[k]) + self.G @ np.nan_to_num(U[k])
            if not np.all(np.isfinite(X[k + 1])):
                X[k + 1] = np.zeros(self.nx)
        return X

    def simulate_matching(
        self, Xc: np.ndarray, Uc: np.ndarray, dt: Optional[float] = None
    ) -> np.ndarray:
        return self.simulate(Xc[0], Uc, dt)[:-1]

    @property
    def spectral_radius(self) -> float:
        if self.F is not None:
            return float(np.max(np.abs(np.linalg.eigvals(self.F))))
        return float(np.max(np.real(np.linalg.eigvals(self.Ac))))

    @property
    def is_stable(self) -> bool:
        return self.spectral_radius <= 1.0

    @staticmethod
    def theta_dim(nu: int) -> int:
        return 10 + 3 * nu

    @staticmethod
    def unpack_theta(
        theta: np.ndarray,
        nu: int,
        u0: float,
        theta0: float,
        w0: float,
        g: float = G_ACC,
    ) -> Tuple[np.ndarray, np.ndarray]:
        Ac = np.zeros((4, 4))
        Ac[0, 0] = theta[0];  Ac[0, 1] = theta[1];  Ac[0, 2] = theta[2]
        Ac[0, 3] = -g * np.cos(theta0)
        Ac[1, 0] = theta[3];  Ac[1, 1] = theta[4];  Ac[1, 2] = theta[5]
        Ac[1, 3] = -g * np.sin(theta0)
        Ac[2, 0] = theta[6];  Ac[2, 1] = theta[7];  Ac[2, 2] = theta[8];  Ac[2, 3] = theta[9]
        Ac[3, :] = [0.0, 0.0, 1.0, 0.0]
        Bc = np.zeros((4, nu))
        for j in range(nu):
            base = 10 + 3 * j
            Bc[0, j] = theta[base];  Bc[1, j] = theta[base + 1];  Bc[2, j] = theta[base + 2]
        return Ac, Bc

    @staticmethod
    def pack_theta(
        Ac: np.ndarray,
        Bc: np.ndarray,
        nu: int,
        theta0: float,
        g: float = G_ACC,
    ) -> np.ndarray:
        theta = np.zeros(StateSpaceModel.theta_dim(nu))
        theta[0] = Ac[0, 0];  theta[1] = Ac[0, 1];  theta[2] = Ac[0, 2]
        theta[3] = Ac[1, 0];  theta[4] = Ac[1, 1];  theta[5] = Ac[1, 2]
        theta[6] = Ac[2, 0];  theta[7] = Ac[2, 1];  theta[8] = Ac[2, 2];  theta[9] = Ac[2, 3]
        for j in range(nu):
            base = 10 + 3 * j
            theta[base] = Bc[0, j];  theta[base + 1] = Bc[1, j];  theta[base + 2] = Bc[2, j]
        return theta

    @staticmethod
    def zoh(
        A: np.ndarray, B: Optional[np.ndarray], h: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        nx = A.shape[0]
        if B is None:
            return expm(A * h), None
        nu = B.shape[1]
        M = np.zeros((nx + nu, nx + nu), float)
        M[:nx, :nx] = A
        M[:nx, nx:] = B
        E = expm(M * h)
        return E[:nx, :nx], E[:nx, nx:]

    def to_theta(self) -> np.ndarray:
        return StateSpaceModel.pack_theta(self.Ac, self.Bc, self.nu, self.theta0, self.g)

    @classmethod
    def from_theta(
        cls,
        theta: np.ndarray,
        nu: int,
        u0: float,
        theta0: float,
        w0: float,
        g: float = G_ACC,
    ) -> "StateSpaceModel":
        Ac, Bc = StateSpaceModel.unpack_theta(theta, nu, u0, theta0, w0, g)
        return cls(Ac, Bc, nu=nu, u0=u0, theta0=theta0, w0=w0, g=g)

    def __repr__(self) -> str:
        return (
            f"StateSpaceModel(nu={self.nu}, stable={self.is_stable}, "
            f"ρ={self.spectral_radius:.4f}, "
            f"θ₀={np.rad2deg(self.theta0):.2f}°, u₀={self.u0:.1f} m/s)"
        )

# ─────────────────────────────────────────────────────────────────────────────
# MOESPIdentifier
# ─────────────────────────────────────────────────────────────────────────────
class MOESPIdentifier:
    def __init__(self, order: int = 4, SS_f: int = 20):
        self.order = order
        self.SS_f  = SS_f

    def fit(
        self,
        Xc: np.ndarray,
        Uc: np.ndarray,
        dt: float,
        u0: float = 0.0,
        theta0: float = 0.0,
        w0: float = 0.0,
    ) -> StateSpaceModel:
        if not _HAS_SIPPY:
            raise ImportError(
                "sippy_unipi is required for MOESPIdentifier. "
                "Install it with: pip install sippy-unipi"
            )

        nu = Uc.shape[1]
        print(f"[TRACE] MOESPIdentifier.fit(): appel _sippy_sid  N={len(Xc)} nu={nu} SS_f={self.SS_f} ...", flush=True)

        sys_id = _sippy_sid(
            Xc, Uc,
            id_method="MOESP",
            SS_f=self.SS_f,
            SS_fixed_order=self.order,
            centering="None",
            tsample=float(dt),
            SS_D_required=False,
            SS_A_stability=False,
        )
        print("[TRACE] MOESPIdentifier.fit(): _sippy_sid terminé", flush=True)

        A_m = np.array(sys_id.A, dtype=float)
        B_m = np.array(sys_id.B, dtype=float)
        C_m = np.array(sys_id.C, dtype=float)

        try:
            T = np.linalg.inv(C_m)
        except np.linalg.LinAlgError:
            T = np.linalg.pinv(C_m)

        A_phys = C_m @ A_m @ T
        B_phys = C_m @ B_m

        try:
            Ac = np.real(logm(A_phys)) / dt
        except Exception:
            Ac = (A_phys - np.eye(self.order)) / dt

        I4 = np.eye(self.order)
        try:
            Bc = np.linalg.solve(A_phys - I4, Ac @ B_phys)
        except np.linalg.LinAlgError:
            Bc = B_phys / dt

        model = StateSpaceModel(Ac, Bc, nu=nu, u0=u0, theta0=theta0, w0=w0)
        model.discretize(dt)
        print("[TRACE] MOESPIdentifier.fit(): fin", flush=True)
        return model

# ─────────────────────────────────────────────────────────────────────────────
# _GreyBoxBase
# ─────────────────────────────────────────────────────────────────────────────
class _GreyBoxBase:
    def __init__(
        self,
        lambda_x: float = 1.0,
        lambda_acc: float = 50.0,
        lambda_reg: float = 0.0,
        max_nfev: int = 50,
        loss: str = "huber",
        f_scale: float = 10.0,
        accel_mode: str = "simple",
        reject_unstable: bool = True,
        unstable_rho_max: float = 1.02,
        reject_bad_xtol: bool = True,
        xtol_max_nfev: int = 5,
        xtol_max_cost: float = 1e8,
        x_clip: float = 1e6,
        bad_penalty: float = 1e6,
        xtol: float = 1e-8,
        ftol: float = 1e-8,
        gtol: float = 1e-8,
        verbose: bool = False,
        downsample: bool = True,
        ds: int = 4,
    ):
        self.lambda_x = lambda_x
        self.lambda_acc = lambda_acc
        self.lambda_reg = lambda_reg
        self.max_nfev = max_nfev
        self.loss = loss
        self.f_scale = f_scale
        self.accel_mode = accel_mode
        self.reject_unstable = reject_unstable
        self.unstable_rho_max = unstable_rho_max
        self.reject_bad_xtol = reject_bad_xtol
        self.xtol_max_nfev = xtol_max_nfev
        self.xtol_max_cost = xtol_max_cost
        self.x_clip = x_clip
        self.bad_penalty = bad_penalty
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.verbose = verbose
        self.downsample = downsample
        self.ds = max(1, int(ds))

    @staticmethod
    def _normalize_triplet(Xc, Uc, Aextra_c):
        Xc = np.asarray(Xc, float)
        x_scale = np.std(Xc, axis=0)
        x_scale = np.where((~np.isfinite(x_scale)) | (x_scale < 1e-6), 1.0, x_scale)
        Xn = Xc / x_scale
        if Uc is not None:
            Uc = np.asarray(Uc, float)
            u_scale = np.std(Uc, axis=0)
            u_scale = np.where((~np.isfinite(u_scale)) | (u_scale < 1e-6), 1.0, u_scale)
            Un = Uc / u_scale
        else:
            u_scale = None
            Un = None
        Aextra_c = np.asarray(Aextra_c, float)
        median = np.median(Aextra_c, axis=0)
        mad = np.median(np.abs(Aextra_c - median), axis=0)
        robust_std = 1.4826 * mad
        robust_std = np.where((~np.isfinite(robust_std)) | (robust_std < 1e-3), 1.0, robust_std)
        a_scale = np.maximum(robust_std, 0.1 * 9.80665)
        Aextra_n = Aextra_c / a_scale
        return Xn, Un, Aextra_n, x_scale, u_scale, a_scale

    @staticmethod
    def _denormalize_matrices(A_n, B_n, x_scale, u_scale):
        Sx     = np.diag(x_scale)
        Sx_inv = np.diag(1.0 / x_scale)
        A_phys = Sx @ A_n @ Sx_inv
        if B_n is None or u_scale is None:
            B_phys = None
        else:
            Su_inv = np.diag(1.0 / u_scale)
            B_phys = Sx @ B_n @ Su_inv
        return A_phys, B_phys

    @staticmethod
    def _to_normalized_matrices(A_phys, B_phys, x_scale, u_scale):
        Sx_inv = np.diag(1.0 / x_scale)
        Sx     = np.diag(x_scale)
        A_n    = Sx_inv @ A_phys @ Sx
        if B_phys is None or u_scale is None:
            B_n = None
        else:
            Su  = np.diag(u_scale)
            B_n = Sx_inv @ B_phys @ Su
        return A_n, B_n

    def fit(
        self,
        model_init: StateSpaceModel,
        Xc: np.ndarray,
        Uc: np.ndarray,
        Acc_c: np.ndarray,
        dt: float,
    ) -> StateSpaceModel:

        nu     = model_init.nu
        u0     = model_init.u0
        theta0 = model_init.theta0
        w0     = model_init.w0
        g0     = model_init.g

        if self.downsample and self.ds > 1:
            Xc_opt    = Xc[::self.ds]
            Uc_opt    = Uc[::self.ds]
            Acc_c_opt = Acc_c[::self.ds]
            dt_opt    = dt * self.ds
        else:
            Xc_opt, Uc_opt, Acc_c_opt, dt_opt = Xc, Uc, Acc_c, dt

        print(f"[TRACE] GreyBox.fit(): démarre least_squares  N_opt={len(Xc_opt)} max_nfev={self.max_nfev} ...", flush=True)
        t0_nls = time.time()

        Xn, Un, _, x_scale, u_scale, a_scale = self._normalize_triplet(Xc_opt, Uc_opt, Acc_c_opt)

        theta_init = self._pack_theta(model_init.Ac, model_init.Bc, nu=nu, theta0=theta0, g0=g0)
        bounds     = self._make_bounds(u0=u0, nu=nu)
        lb, ub     = bounds

        theta_init = np.asarray(theta_init, dtype=float)
        lb         = np.asarray(lb, dtype=float)
        ub         = np.asarray(ub, dtype=float)

        theta_init = np.clip(theta_init, lb, ub)
        eps        = 1e-12
        theta_init = np.minimum(theta_init, ub - eps)
        theta_init = np.maximum(theta_init, lb + eps)

        theta_ref = theta_init.copy()
        reg = None
        if self.lambda_reg is not None and self.lambda_reg > 0:
            reg = (theta_ref, self.lambda_reg)

        def fun(theta):
            return self._residuals_classic(
                theta=theta, Xn=Xn, Un=Un, Acc_c=Acc_c_opt, dt=dt_opt,
                x_scale=x_scale, u_scale=u_scale, a_scale=a_scale,
                u0=u0, theta0=theta0, w0=w0, g0=g0, nu=nu, reg=reg,
            )

        res = least_squares(
            fun=fun, x0=theta_init, method="trf", bounds=(lb, ub),
            loss=self.loss, f_scale=self.f_scale, max_nfev=self.max_nfev,
            xtol=self.xtol, ftol=self.ftol, gtol=self.gtol,
            x_scale="jac", verbose=0,
        )
        print(f"[TRACE] GreyBox.fit(): least_squares terminé en {time.time()-t0_nls:.1f}s, nfev={res.nfev}, status={res.status}", flush=True)

        if self.verbose:
            status_dict = {
                -1: "improper input parameters",
                 0: "max function evaluations reached",
                 1: "gtol satisfied", 2: "ftol satisfied",
                 3: "xtol satisfied", 4: "both ftol and xtol satisfied",
            }
            print("\n🔎 ===== LSQ DEBUG =====")
            print(f"Status code   : {res.status}")
            print(f"Reason        : {status_dict.get(res.status, 'unknown')}")
            print(f"Message       : {res.message}")
            print(f"Iterations    : {res.nfev}")
            print(f"Final cost    : {res.cost:.6e}")
            print(f"Success       : {res.success}")
            print("================================\n")

        if self.reject_bad_xtol:
            bad_xtol = (
                res.status == 3
                and (
                    res.nfev <= self.xtol_max_nfev
                    or not np.isfinite(res.cost)
                    or res.cost > self.xtol_max_cost
                )
            )
            if bad_xtol:
                raise RuntimeError(
                    f"Faux succès xtol : status={res.status}, "
                    f"nfev={res.nfev}, cost={res.cost:.3e}"
                )

        A_n, B_n = self._unpack_theta_normalized(
            res.x, nu=nu, u0=u0, theta0=theta0, w0=w0,
            x_scale=x_scale, u_scale=u_scale[:nu], g0=g0,
        )

        Ac_opt, Bc_opt = self._denormalize_matrices(A_n, B_n, x_scale=x_scale, u_scale=u_scale[:nu])

        model_out = StateSpaceModel(Ac_opt, Bc_opt, nu=nu, u0=u0, theta0=theta0, w0=w0, g=g0)
        model_out.discretize(dt)

        rhoF = model_out.spectral_radius
        if self.verbose:
            print(f"📈 Spectral radius(F) = {rhoF:.6f}")

        if self.reject_unstable:
            if not np.isfinite(rhoF) or rhoF > self.unstable_rho_max:
                raise RuntimeError(f"Modèle instable après fit : spectral radius(F) = {rhoF:.6f}")

        Xsim_c = model_out.simulate_matching(Xc, Uc[:, :nu], dt)
        if not np.all(np.isfinite(Xsim_c)):
            raise RuntimeError("Simulation libre non finie après fit.")

        max_abs_xsim = float(np.max(np.abs(Xsim_c)))
        if self.verbose:
            print(f"📉 max|Xsim_c| = {max_abs_xsim:.6e}")
        if not np.isfinite(max_abs_xsim) or max_abs_xsim > self.x_clip:
            raise RuntimeError(f"Simulation libre divergente : max|Xsim_c| = {max_abs_xsim:.6e}")

        Cextra, Dextra = self._build_output_matrices(Ac_opt, Bc_opt, mode=self.accel_mode)

        if Uc is None or nu == 0:
            Acc_pred = (Cextra @ Xsim_c.T).T
        else:
            Acc_pred = (Cextra @ Xsim_c.T).T + (Dextra @ Uc[:, :nu].T).T

        rmse_accel = float(np.sqrt(np.mean((Acc_c - Acc_pred) ** 2)))

        Xk          = Xc[:-1]
        Xkp1        = Xc[1:]
        Uk          = Uc[:-1, :nu]
        Xpred_1step = (model_out.F @ Xk.T + model_out.G @ Uk.T).T
        rmse_1step  = float(np.sqrt(np.mean((Xkp1 - Xpred_1step) ** 2)))

        model_out._ls_result     = res
        model_out._fit_bounds      = bounds
        model_out._fit_param_names = self._param_names_classic(nu)
        model_out._fit_has_reg      = reg is not None
        model_out._x_scale    = x_scale
        model_out._u_scale    = u_scale
        model_out._a_scale    = a_scale
        model_out._Cextra     = Cextra
        model_out._Dextra     = Dextra
        model_out._rmse_accel = rmse_accel
        model_out._rmse_1step = rmse_1step
        model_out._n_fit      = int(Xc_opt.shape[0])
        return model_out

    def _residuals_classic(
        self,
        theta: np.ndarray,
        Xn: np.ndarray,
        Un: np.ndarray,
        Acc_c: np.ndarray,
        dt: float,
        x_scale: np.ndarray,
        u_scale: np.ndarray,
        a_scale: np.ndarray,
        u0: float,
        theta0: float,
        w0: float,
        g0: float,
        nu: int,
        reg=None,
    ) -> np.ndarray:

        N, nx = Xn.shape
        na    = Acc_c.shape[1]

        if Un is None:
            U_use_n    = None
            U_use_phys = None
        else:
            U_use_n    = Un[:, :nu]
            U_use_phys = U_use_n * u_scale[:nu]

        A_n, B_n = self._unpack_theta_normalized(
            theta, nu=nu, u0=u0, theta0=theta0, w0=w0,
            x_scale=x_scale, u_scale=u_scale[:nu], g0=g0,
        )

        A_phys, B_phys = self._denormalize_matrices(A_n, B_n, x_scale=x_scale, u_scale=u_scale[:nu])

        F_n, G_n = StateSpaceModel.zoh(A_n, B_n, dt)

        Xsim_n    = np.zeros_like(Xn)
        Xsim_n[0] = Xn[0]

        for k in range(N - 1):
            x_next = F_n @ Xsim_n[k] if U_use_n is None else F_n @ Xsim_n[k] + G_n @ U_use_n[k]

            if not np.all(np.isfinite(x_next)) or np.max(np.abs(x_next)) > self.x_clip:
                r_bad = self.bad_penalty * np.ones((N, nx + na), dtype=float).ravel()
                if reg is not None:
                    theta_ref, lam = reg
                    r_bad = np.concatenate([r_bad, np.sqrt(lam) * (theta - theta_ref)])
                return r_bad

            Xsim_n[k + 1] = x_next

        err_x = Xn - Xsim_n
        r_x   = np.sqrt(self.lambda_x) * err_x.ravel()

        Xsim_phys      = Xsim_n * x_scale.reshape(1, -1)
        Cextra, Dextra = self._build_output_matrices(A_phys, B_phys, mode=self.accel_mode)

        if U_use_phys is None:
            Acc_pred = (Cextra @ Xsim_phys.T).T
        else:
            Acc_pred = (Cextra @ Xsim_phys.T).T + (Dextra @ U_use_phys.T).T

        err_a = np.tanh((Acc_c - Acc_pred) / a_scale.reshape(1, -1))
        r_a   = np.sqrt(self.lambda_acc) * err_a.ravel()

        if not np.all(np.isfinite(r_x)) or not np.all(np.isfinite(r_a)):
            r_bad = self.bad_penalty * np.ones((N, nx + na), dtype=float).ravel()
            if reg is not None:
                theta_ref, lam = reg
                r_bad = np.concatenate([r_bad, np.sqrt(lam) * (theta - theta_ref)])
            return r_bad

        r = np.concatenate([r_x, r_a])
        if reg is not None:
            theta_ref, lam = reg
            r = np.concatenate([r, np.sqrt(lam) * (theta - theta_ref)])
        return r

# ─────────────────────────────────────────────────────────────────────────────
# LongitudinalGreyBoxRefinement
# ─────────────────────────────────────────────────────────────────────────────
class LongitudinalGreyBoxRefinement(_GreyBoxBase):
    @staticmethod
    def theta_dim_classic(nu: int) -> int:
        nu = int(nu)
        if nu < 1:
            raise ValueError("nu doit être >= 1")
        return 13 + 3 * (nu - 1)

    @staticmethod
    def _unpack_theta(theta, nu, u0, theta0, w0, g0=9.80665):
        th  = np.asarray(theta, float).ravel()
        nu  = int(nu)
        if nu < 1:
            raise ValueError("nu doit être >= 1")
        expected = 13 + 3 * (nu - 1)
        if len(th) < expected:
            raise ValueError(f"theta trop court: attendu {expected}, reçu {len(th)}")
        if not (np.isfinite(u0) and u0 > 0):
            raise ValueError(f"u0 invalide (>0 et fini). Reçu: {u0}")
        if not np.isfinite(theta0):
            raise ValueError(f"theta0 invalide (fini). Reçu: {theta0}")
        if not np.isfinite(w0):
            raise ValueError(f"w0 invalide (fini). Reçu: {w0}")
        cg  = float(np.cos(theta0))
        sg  = float(np.sin(theta0))
        A   = np.zeros((4, 4), float)
        B   = np.zeros((4, nu), float)
        idx = 0
        Xu, Xw, Xq           = th[idx:idx+3]; idx += 3
        Zu, Zw, Zq           = th[idx:idx+3]; idx += 3
        Mu, Mw, Mq, Mtheta   = th[idx:idx+4]; idx += 4
        A[0, 0] = Xu;  A[0, 1] = Xw;  A[0, 2] = -w0;  A[0, 3] = -g0 * cg
        A[1, 0] = Zu;  A[1, 1] = Zw;  A[1, 2] =  u0;  A[1, 3] = -g0 * sg
        A[2, 0] = Mu;  A[2, 1] = Mw;  A[2, 2] =  Mq;  A[2, 3] = 0.0
        A[3, :] = 0.0; A[3, 2] = 1.0
        for k in range(nu):
            Xdu, Zdu, Mdu = th[idx:idx+3]; idx += 3
            B[0, k] = Xdu;  B[1, k] = Zdu;  B[2, k] = Mdu;  B[3, k] = 0.0
        return A, B

    @classmethod
    def _unpack_theta_normalized(cls, theta, nu, u0, theta0, w0, x_scale, u_scale, g0=9.80665):
        A_phys, B_phys = cls._unpack_theta(theta, nu, u0=u0, theta0=theta0, w0=w0, g0=g0)
        A_n, B_n       = cls._to_normalized_matrices(A_phys, B_phys, x_scale, u_scale)
        return A_n, B_n

    @classmethod
    def _pack_theta(cls, Ac, Bc, nu, theta0=0.0, g0=9.80665):
        nu = int(nu)
        if nu < 1:
            raise ValueError("nu doit être >= 1")
        if Bc.shape[1] != nu:
            raise ValueError(f"Bc doit être (4,nu). Reçu {Bc.shape}, nu={nu}")
        th  = np.zeros(cls.theta_dim_classic(nu), float)
        idx = 0
        th[idx:idx+3] = [Ac[0, 0], Ac[0, 1], Ac[0, 2]]; idx += 3
        th[idx:idx+3] = [Ac[1, 0], Ac[1, 1], Ac[1, 2]]; idx += 3
        th[idx:idx+4] = [Ac[2, 0], Ac[2, 1], Ac[2, 2], Ac[2, 3]]; idx += 4
        for k in range(nu):
            th[idx:idx+3] = [Bc[0, k], Bc[1, k], Bc[2, k]]; idx += 3
        return th

    @classmethod
    def _param_names_classic(cls, nu: int) -> List[str]:
        nu = int(nu)
        names = ["Xu", "Xw", "Xq", "Zu", "Zw", "Zq", "Mu", "Mw", "Mq", "Mtheta"]
        input_tags = ["de", "n1"]
        for k in range(nu):
            tag = input_tags[k] if k < len(input_tags) else f"in{k}"
            names += [f"X{tag}", f"Z{tag}", f"M{tag}"]
        return names

    @staticmethod
    def _make_bounds(u0, nu=1):
        u0  = float(u0)
        nu  = int(nu)
        if not (np.isfinite(u0) and u0 > 0):
            raise ValueError(f"u0 invalide: {u0}")
        if nu < 1:
            raise ValueError("nu doit être >= 1")
        dim = 10 + 3 * nu
        lb  = np.full(dim, -np.inf, dtype=float)
        ub  = np.full(dim,  np.inf, dtype=float)
        lb[0],  ub[0]  = -1.0,    0.0
        lb[1],  ub[1]  =  0.0,   10.0
        lb[3],  ub[3]  = -1.0,    0.0
        lb[4],  ub[4]  = -5.0,    0.0
        lb[6],  ub[6]  = -0.5,    0.5
        lb[7],  ub[7]  = -5.0,    0.0
        lb[8],  ub[8]  = -40.0,   0.0
        eps_fix = 1e-12
        lb[2],  ub[2]  = -eps_fix, eps_fix
        lb[5],  ub[5]  = -eps_fix, eps_fix
        lb[9],  ub[9]  = -eps_fix, eps_fix
        for j in range(nu):
            i0 = 10 + 3 * j
            if j == 0:
                lb[i0+0], ub[i0+0] = -eps_fix, eps_fix
                lb[i0+1], ub[i0+1] = -1000,  0
                lb[i0+2], ub[i0+2] = -100.0,  -0.1
            elif j == 1:
                lb[i0+0], ub[i0+0] =  0.01,  50.0
                lb[i0+1], ub[i0+1] = -0.5,    0.5
                lb[i0+2], ub[i0+2] =  0.0,   20.0
        return lb, ub

    @staticmethod
    def _build_output_matrices(A_c, B_c, mode="simple"):
        A_c = np.asarray(A_c, float)
        B_c = np.asarray(B_c, float)
        if A_c.shape != (4, 4):
            raise ValueError(f"A_c doit être 4x4, reçu {A_c.shape}")
        if B_c.ndim != 2 or B_c.shape[0] != 4:
            raise ValueError(f"B_c doit être (4,nu), reçu {B_c.shape}")
        if mode == "simple":
            Cextra = np.array([
                [A_c[0, 0], A_c[0, 1], 0.0, 0.0],
                [A_c[1, 0], A_c[1, 1], 0.0, 0.0],
            ], dtype=float)
        elif mode == "full":
            Cextra = np.array([
                [A_c[0, 0], A_c[0, 1], A_c[0, 2], A_c[0, 3]],
                [A_c[1, 0], A_c[1, 1], A_c[1, 2], A_c[1, 3]],
            ], dtype=float)
        else:
            raise ValueError("mode doit être 'simple' ou 'full'")
        Dextra = np.array([B_c[0, :], B_c[1, :]], dtype=float)
        return Cextra, Dextra

# ─────────────────────────────────────────────────────────────────────────────
# LateralGreyBoxRefinement
# ─────────────────────────────────────────────────────────────────────────────
class LateralGreyBoxRefinement(_GreyBoxBase):
    @staticmethod
    def theta_dim_classic(nu: int) -> int:
        nu = int(nu)
        if nu < 1:
            raise ValueError("nu doit être >= 1")
        return 9 + 3 * nu

    @staticmethod
    def _unpack_theta(theta, nu, u0, theta0, w0, g0=9.80665):
        th  = np.asarray(theta, float).ravel()
        nu  = int(nu)
        if not (np.isfinite(u0) and u0 > 0):
            raise ValueError(f"u0 invalide: {u0}")
        if not np.isfinite(theta0):
            raise ValueError(f"theta0 invalide: {theta0}")
        if not np.isfinite(w0):
            raise ValueError(f"w0 invalide: {w0}")
        A   = np.zeros((4, 4), float)
        B   = np.zeros((4, nu), float)
        idx = 0
        Yv, _Yp, _Yr = th[idx:idx+3]; idx += 3
        Lv, Lp, Lr   = th[idx:idx+3]; idx += 3
        Nv, Np, Nr   = th[idx:idx+3]; idx += 3
        A[0, 0] = Yv
        A[0, 1] = w0
        A[0, 2] = -u0
        A[0, 3] = g0 * np.cos(theta0)
        A[1, 0] = Lv;  A[1, 1] = Lp;  A[1, 2] = Lr;  A[1, 3] = 0.0
        A[2, 0] = Nv;  A[2, 1] = Np;  A[2, 2] = Nr;  A[2, 3] = 0.0
        A[3, :] = [0.0, 1.0, np.tan(theta0), 0.0]
        for k in range(nu):
            Ydu, Ldu, Ndu = th[idx:idx+3]; idx += 3
            B[0, k] = Ydu;  B[1, k] = Ldu;  B[2, k] = Ndu;  B[3, k] = 0.0
        return A, B

    @classmethod
    def _unpack_theta_normalized(cls, theta, nu, u0, theta0, w0, x_scale, u_scale, g0=9.80665):
        A_phys, B_phys = cls._unpack_theta(theta, nu, u0=u0, theta0=theta0, w0=w0, g0=g0)
        A_n, B_n       = cls._to_normalized_matrices(A_phys, B_phys, x_scale, u_scale)
        return A_n, B_n

    @classmethod
    def _pack_theta(cls, Ac, Bc, nu, theta0=0.0, g0=9.80665):
        nu  = int(nu)
        th  = np.zeros(cls.theta_dim_classic(nu), float)
        idx = 0
        th[idx:idx+3] = [Ac[0, 0], Ac[0, 1], Ac[0, 2]]; idx += 3
        th[idx:idx+3] = [Ac[1, 0], Ac[1, 1], Ac[1, 2]]; idx += 3
        th[idx:idx+3] = [Ac[2, 0], Ac[2, 1], Ac[2, 2]]; idx += 3
        for k in range(nu):
            th[idx:idx+3] = [Bc[0, k], Bc[1, k], Bc[2, k]]; idx += 3
        return th

    @classmethod
    def _param_names_classic(cls, nu: int) -> List[str]:
        nu = int(nu)
        names = ["Yv", "Yp", "Yr", "Lv", "Lp", "Lr", "Nv", "Np", "Nr"]
        input_tags = ["da", "dr"]
        for k in range(nu):
            tag = input_tags[k] if k < len(input_tags) else f"in{k}"
            names += [f"Y{tag}", f"L{tag}", f"N{tag}"]
        return names

    @staticmethod
    def _make_bounds(u0, nu=2):
        nu  = int(nu)
        dim = 9 + 3 * nu
        lb  = np.full(dim, -np.inf, dtype=float)
        ub  = np.full(dim,  np.inf, dtype=float)
        lb[0], ub[0] = -5.0,   0.0
        lb[1], ub[1] = -500.0, 500.0
        lb[2], ub[2] = -500.0, 500.0
        lb[3], ub[3] = -5.0,   5.0
        lb[4], ub[4] = -50.0,  0.0
        lb[5], ub[5] = -5.0,   10.0
        lb[6], ub[6] = -5.0,   5.0
        lb[7], ub[7] = -5.0,   5.0
        lb[8], ub[8] = -20.0,  0.0
        for j in range(nu):
            i0 = 9 + 3 * j
            if j == 0:
                lb[i0+0], ub[i0+0] = -50.0,  50.0
                lb[i0+1], ub[i0+1] = -200.0, 200.0
                lb[i0+2], ub[i0+2] = -200.0, 200.0
            elif j == 1:
                lb[i0+0], ub[i0+0] = -50.0,  50.0
                lb[i0+1], ub[i0+1] = -100.0, 100.0
                lb[i0+2], ub[i0+2] = -100.0, 100.0
        return lb, ub

    @staticmethod
    def _build_output_matrices(A_c, B_c, mode="simple"):
        A_c = np.asarray(A_c, float)
        B_c = np.asarray(B_c, float)
        if mode == "simple":
            Cextra = np.array([[A_c[0, 0], 0.0, 0.0, 0.0]], dtype=float)
        elif mode == "full":
            Cextra = A_c[0:1, :].copy()
        else:
            raise ValueError("mode doit être 'simple' ou 'full'")
        Dextra = B_c[0:1, :].copy()
        return Cextra, Dextra

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Metrics:
    r2:      np.ndarray
    pearson: np.ndarray
    fit_pct: np.ndarray
    rmse:    np.ndarray
    dynamics: str
    derivatives: Dict[str, float] = field(default_factory=dict)
    trim_kde: Dict[str, float] = field(default_factory=dict)
    maneuver_2311: Dict = field(default_factory=dict)
    aero: Dict[str, float] = field(default_factory=dict)
    excitation: Dict[str, float] = field(default_factory=dict)
    fisher_info: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        Xc:     np.ndarray,
        Xsim:   np.ndarray,
        model:  "StateSpaceModel",
        config: "AircraftConfig",
        trim:   "TrimResult",
        dt:     float,
    ) -> "Metrics":
        r2, pearson, fit_pct, rmse = cls._compute_reconstruction(Xc, Xsim)
        derivatives = cls._extract_derivatives(model, config)
        trim_kde    = cls._summarize_kde_scores(trim, dt) if trim.scores_df is not None else {}
        return cls(
            r2=r2,
            pearson=pearson,
            fit_pct=fit_pct,
            rmse=rmse,
            dynamics=config.dynamics,
            derivatives=derivatives,
            trim_kde=trim_kde,
            maneuver_2311={},
        )

    @staticmethod
    def _compute_reconstruction(
        Y_meas: np.ndarray,
        Y_hat:  np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ny = Y_meas.shape[1]
        r2_arr, pear_arr, fit_arr, rmse_arr = (np.zeros(ny) for _ in range(4))
        for i in range(ny):
            ym, yh = Y_meas[:, i], Y_hat[:, i]
            mask = np.isfinite(ym) & np.isfinite(yh)
            if mask.sum() < 2:
                continue
            ym_m, yh_m = ym[mask], yh[mask]
            r2_arr[i]   = float(r2_score(ym_m, yh_m))
            pr, _       = pearsonr(ym_m, yh_m)
            pear_arr[i] = float(pr) if np.isfinite(pr) else 0.0
            denom       = np.linalg.norm(ym_m - ym_m.mean())
            fit_arr[i]  = float(1.0 - np.linalg.norm(ym_m - yh_m) / denom) if denom > 1e-12 else 0.0
            rmse_arr[i] = float(np.sqrt(np.mean((ym_m - yh_m) ** 2)))
        return r2_arr, pear_arr, fit_arr, rmse_arr

    @staticmethod
    def _aero_signals(
        Xc:       np.ndarray,
        Xsim:     np.ndarray,
        dynamics: str,
        u0:       float,
        w0:       float,
        v0:       float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N = len(Xc)
        if dynamics == "lateral":
            u_m = np.full(N, u0);  w_m = np.full(N, w0);  v_m = Xc[:, 0] + v0
            u_s = np.full(N, u0);  w_s = np.full(N, w0);  v_s = Xsim[:, 0] + v0
        elif dynamics == "six_dof":
            u_m = Xc[:, 0] + u0;  v_m = Xc[:, 1] + v0;  w_m = Xc[:, 2] + w0
            u_s = Xsim[:, 0] + u0;  v_s = Xsim[:, 1] + v0;  w_s = Xsim[:, 2] + w0
        else:
            u_m = Xc[:, 0] + u0;  w_m = Xc[:, 1] + w0;  v_m = np.full(N, v0)
            u_s = Xsim[:, 0] + u0;  w_s = Xsim[:, 1] + w0;  v_s = np.full(N, v0)

        TAS_m   = np.sqrt(u_m**2 + v_m**2 + w_m**2)
        TAS_s   = np.sqrt(u_s**2 + v_s**2 + w_s**2)
        alpha_m = np.arctan2(w_m, u_m)
        alpha_s = np.arctan2(w_s, u_s)
        beta_m  = np.arctan2(v_m, np.sqrt(u_m**2 + w_m**2))
        beta_s  = np.arctan2(v_s, np.sqrt(u_s**2 + w_s**2))
        return TAS_m, alpha_m, beta_m, TAS_s, alpha_s, beta_s

    @staticmethod
    def _compute_aero_metrics(
        Xc:       np.ndarray,
        Xsim:     np.ndarray,
        dynamics: str,
        u0:       float,
        w0:       float,
        v0:       float = 0.0,
    ) -> Dict[str, float]:
        TAS_m, alpha_m, beta_m, TAS_s, alpha_s, beta_s = Metrics._aero_signals(
            Xc, Xsim, dynamics, u0, w0, v0
        )
        if dynamics == "lateral":
            channels = [(beta_m,  beta_s,  "beta")]
        elif dynamics == "six_dof":
            channels = [(TAS_m,   TAS_s,   "tas"),
                        (alpha_m, alpha_s, "alpha"),
                        (beta_m,  beta_s,  "beta")]
        else:
            channels = [(TAS_m,   TAS_s,   "tas"),
                        (alpha_m, alpha_s, "alpha")]

        out: Dict[str, float] = {}
        for ym, ys, name in channels:
            Y_meas = ym.reshape(-1, 1)
            Y_hat  = ys.reshape(-1, 1)
            r2, pearson, fit_pct, rmse = Metrics._compute_reconstruction(Y_meas, Y_hat)
            out[f"{name}_r2"]      = float(r2[0])
            out[f"{name}_pearson"] = float(pearson[0])
            out[f"{name}_fit_pct"] = float(fit_pct[0])
            out[f"{name}_rmse"]    = float(rmse[0])
        return out

    @staticmethod
    def _state_input_names(dynamics: str, n_u: int) -> Tuple[List[str], List[str]]:
        state_names = (
            ["v", "p", "r", "phi"]                              if dynamics == "lateral"
            else ["u", "v", "w", "p", "q", "r", "phi", "theta"] if dynamics == "six_dof"
            else ["u", "w", "q", "theta"]
        )
        input_names = (
            ["da", "dr"]      if dynamics == "lateral"
            else [f"in{j}" for j in range(n_u)] if dynamics == "six_dof"
            else ["de", "n1"]
        )
        return state_names, input_names

    @staticmethod
    def _compute_excitation(Xc: np.ndarray, Uc: np.ndarray,
                             dynamics: str = "longitudinal") -> Dict[str, float]:
        n = min(len(Xc), len(Uc))
        if n < 2:
            return {}
        X = np.asarray(Xc[:n], dtype=float)
        U = np.asarray(Uc[:n], dtype=float)
        x_std = np.std(X, axis=0)
        u_std = np.std(U, axis=0)
        x_scale = np.where((~np.isfinite(x_std)) | (x_std < 1e-9), 1.0, x_std)
        u_scale = np.where((~np.isfinite(u_std)) | (u_std < 1e-9), 1.0, u_std)
        phi = np.concatenate([X / x_scale, U / u_scale], axis=1)

        mask = np.all(np.isfinite(phi), axis=1)
        phi = phi[mask]
        Xm, Um = X[mask], U[mask]
        if len(phi) < phi.shape[1]:
            return {"excitation_status": "insufficient_samples"}

        R_phi = (phi.T @ phi) / len(phi)
        eigvals = np.linalg.eigvalsh(R_phi)
        lam_min = float(eigvals[0])
        lam_max = float(eigvals[-1])

        state_names, input_names = Metrics._state_input_names(dynamics, n_u=Um.shape[1])
        amplitudes: Dict[str, float] = {}
        for i in range(Xm.shape[1]):
            name = state_names[i] if i < len(state_names) else f"x{i}"
            amplitudes[f"excitation_amp_x_{name}"] = float(np.ptp(Xm[:, i]))
        for j in range(Um.shape[1]):
            name = input_names[j] if j < len(input_names) else f"u{j}"
            amplitudes[f"excitation_amp_u_{name}"] = float(np.ptp(Um[:, j]))

        return {
            **amplitudes,
            "excitation_lambda_min":      lam_min,
            "excitation_lambda_max":      lam_max,
            "excitation_kappa":           lam_max / lam_min if lam_min > 1e-12 else float("inf"),
            "excitation_rank_deficient":  bool(lam_min < 1e-8),
            "excitation_n_regressors":    int(phi.shape[1]),
        }

    @staticmethod
    def _compute_fisher_info(model: "StateSpaceModel", dynamics: Optional[str] = None) -> Dict[str, float]:
        res        = getattr(model, "_ls_result", None)
        bounds     = getattr(model, "_fit_bounds", None)
        names_full = getattr(model, "_fit_param_names", None)
        if res is None or bounds is None or names_full is None:
            return {}

        J = np.asarray(res.jac, dtype=float)
        f = np.asarray(res.fun, dtype=float)
        lb, ub = (np.asarray(b, dtype=float) for b in bounds)
        n_res, n_theta = J.shape
        if n_theta != len(names_full):
            return {}

        if getattr(model, "_fit_has_reg", False) and n_res > n_theta:
            J = J[:-n_theta, :]
            f = f[:-n_theta]
        n_res_data = J.shape[0]

        col_norm    = np.linalg.norm(J, axis=0)
        free_bounds = (ub - lb) > 1e-6
        free_jac    = col_norm > (col_norm.max() * 1e-8 if col_norm.max() > 0 else 0.0)
        free_mask   = free_bounds & free_jac
        n_free = int(free_mask.sum())
        if n_free < 1:
            return {"fisher_status": "no_free_parameters"}

        names  = [n for n, f_name in zip(names_full, free_mask) if f_name]
        J_free = J[:, free_mask]
        dof    = n_res_data - n_free
        if dof < 1:
            return {"fisher_status": "insufficient_dof"}

        FIM = J_free.T @ J_free
        eigvals, eigvecs = np.linalg.eigh(FIM)
        lam_min = float(eigvals[0])
        lam_max = float(eigvals[-1])

        weak_vec   = eigvecs[:, 0]
        weak_terms = sorted(zip(names, weak_vec), key=lambda t: -abs(t[1]))
        weak_direction = " ".join(f"{c:+.2f}*{n}" for n, c in weak_terms if abs(c) > 0.05)

        cost = 0.5 * float(np.sum(f ** 2))
        s_sq = float(2.0 * cost / dof)
        cov  = s_sq * np.linalg.pinv(FIM)
        std  = np.sqrt(np.clip(np.diag(cov), 0.0, None))

        eps_d   = lam_max * 1e-12 if lam_max > 0 else 1e-12
        j_d     = float(np.sum(np.log(eigvals + eps_d)))
        tr_cov  = float(np.trace(cov))
        j_a     = (1.0 / tr_cov) if tr_cov > 1e-300 else float("inf")
        j_kappa = (lam_min / lam_max) if lam_max > 1e-300 else 0.0

        out: Dict[str, float] = {
            "fisher_cond":            (lam_max / lam_min) if lam_min > 1e-300 else float("inf"),
            "fisher_rank_deficient":  bool(lam_min < lam_max * 1e-10),
            "fisher_n_free_params":   n_free,
            "fisher_j_d":             j_d,
            "fisher_j_e":             lam_min,
            "fisher_j_a":             j_a,
            "fisher_j_kappa":         j_kappa,
            "fisher_weak_direction":  weak_direction,
        }
        for name, s in zip(names, std):
            out[f"fisher_std_{name}"] = float(s)

        x_scale = getattr(model, "_x_scale", None)
        u_scale = getattr(model, "_u_scale", None)
        if dynamics is not None and x_scale is not None and u_scale is not None:
            state_names, input_names = Metrics._state_input_names(dynamics, n_u=len(u_scale))
            suffix_to_std = {nm: x_scale[i] for i, nm in enumerate(state_names) if i < len(x_scale)}
            suffix_to_std.update({nm: u_scale[j] for j, nm in enumerate(input_names) if j < len(u_scale)})
            for name in names:
                rs = suffix_to_std.get(name[1:])
                if rs is not None:
                    out[f"fisher_regressor_std_{name}"] = float(rs)

        theta_val = np.asarray(res.x, dtype=float)[free_mask]

        worst_i, worst_rel = None, -1.0
        worst_z_i, worst_z = None, float("inf")
        for i in range(n_free):
            s_i = std[i] if std[i] > 1e-300 else 1e-300
            z_i = abs(theta_val[i]) / s_i
            out[f"fisher_z_{names[i]}"] = float(z_i)
            rel = std[i] / abs(theta_val[i]) if abs(theta_val[i]) > 1e-9 else float("inf")
            if rel > worst_rel:
                worst_rel, worst_i = rel, i
            if z_i < worst_z:
                worst_z, worst_z_i = z_i, i
        out["fisher_worst_param"]    = names[worst_i]
        out["fisher_worst_rel_std"]  = float(worst_rel)
        out["fisher_worst_z_param"]  = names[worst_z_i]
        out["fisher_worst_z"]        = float(worst_z)

        span              = np.maximum(ub[free_mask] - lb[free_mask], 1e-12)
        distance_to_bound = np.minimum(theta_val - lb[free_mask], ub[free_mask] - theta_val) / span
        bound_active      = distance_to_bound < 1e-3
        for name, active in zip(names, bound_active):
            out[f"fisher_bound_active_{name}"] = bool(active)
        out["fisher_any_bound_active"] = bool(np.any(bound_active))

        denom      = np.outer(std, std)
        denom_safe = np.where(denom > 1e-300, denom, 1.0)
        corr = cov / denom_safe
        for i in range(n_free):
            for j in range(i + 1, n_free):
                out[f"fisher_corr_{names[i]}_{names[j]}"] = float(corr[i, j])

        return out

    @staticmethod
    def _extract_derivatives(model: "StateSpaceModel", config: "AircraftConfig") -> Dict[str, float]:
        Ac, Bc = model.Ac, model.Bc
        if config.dynamics == "lateral":
            d: Dict[str, float] = {
                "Yv": Ac[0, 0],
                "Lv": Ac[1, 0], "Lp": Ac[1, 1], "Lr": Ac[1, 2],
                "Nv": Ac[2, 0], "Np": Ac[2, 1], "Nr": Ac[2, 2],
                "Yda": Bc[0, 0], "Lda": Bc[1, 0], "Nda": Bc[2, 0],
            }
            if Bc.shape[1] > 1:
                d.update({"Ydr": Bc[0, 1], "Ldr": Bc[1, 1], "Ndr": Bc[2, 1]})
        elif config.dynamics == "six_dof":
            d = {}
            for i in range(Ac.shape[0]):
                for j in range(Ac.shape[1]):
                    d[f"A{i+1}{j+1}"] = float(Ac[i, j])
            for i in range(Bc.shape[0]):
                for j in range(Bc.shape[1]):
                    d[f"B{i+1}{j+1}"] = float(Bc[i, j])
        else:
            d = {
                "Xu": Ac[0, 0], "Xw": Ac[0, 1],
                "Zu": Ac[1, 0], "Zw": Ac[1, 1],
                "Mu": Ac[2, 0], "Mw": Ac[2, 1], "Mq": Ac[2, 2], "Mtheta": Ac[2, 3],
                "Xde": Bc[0, 0], "Zde": Bc[1, 0], "Mde": Bc[2, 0],
            }
            if Bc.shape[1] > 1:
                d.update({"Xn1": Bc[0, 1], "Zn1": Bc[1, 1], "Mn1": Bc[2, 1]})
        return {k: float(v) for k, v in d.items()}

    @staticmethod
    def _summarize_kde_scores(
        trim: "TrimResult",
        dt: float,
        y_ref: float = 0.95,
        thresholds: Tuple[float, ...] = (0.95, 0.90),
    ) -> Dict[str, float]:
        scores_df = trim.scores_df
        if scores_df is None or len(scores_df) == 0:
            return {}

        try:
            scores_trim = scores_df.iloc[int(trim.i0_avg):int(trim.i1_avg)].copy()
        except Exception:
            return {}

        num_cols = [c for c in scores_trim.columns
                    if np.issubdtype(scores_trim[c].dtype, np.number)]
        out: Dict[str, float] = {}

        for col in num_cols:
            x = pd.to_numeric(scores_trim[col], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue

            prefix = f"trim_{col}"
            out[f"{prefix}_mean"] = float(np.mean(x))
            out[f"{prefix}_min"]  = float(np.min(x))
            out[f"{prefix}_max"]  = float(np.max(x))
            out[f"{prefix}_std"]  = float(np.std(x))

            for th in thresholds:
                tag = str(th).replace(".", "")
                out[f"{prefix}_frac_ge_{tag}"] = float(np.mean(x >= th))

            deficit      = np.maximum(0.0, y_ref - x)
            deficit_area = float(np.sum(deficit) * dt)
            rect_area    = float(y_ref * len(x) * dt)
            area_score   = (
                float(np.clip(1.0 - deficit_area / rect_area, 0.0, 1.0))
                if rect_area > 0 else np.nan
            )
            out[f"{prefix}_area_score"] = area_score

        return out

    @property
    def global_score(self) -> float:
        per_ch = (
            np.clip(self.r2,      0, 1)
            + np.clip(self.pearson, 0, 1)
            + np.clip(self.fit_pct, 0, 1)
        ) / 3.0
        return float(per_ch.mean())

    def summary(self) -> Dict:
        state_names = (
            ["v", "p", "r", "phi"]                              if self.dynamics == "lateral"
            else ["u", "v", "w", "p", "q", "r", "phi", "theta"] if self.dynamics == "six_dof"
            else ["u", "w", "q", "theta"]
        )
        d: Dict = {"global_score": self.global_score}

        for i, sn in enumerate(state_names):
            d[f"r2_{sn}"]      = float(self.r2[i])      if i < len(self.r2)      else np.nan
            d[f"pearson_{sn}"] = float(self.pearson[i]) if i < len(self.pearson) else np.nan
            d[f"fit_pct_{sn}"] = float(self.fit_pct[i]) if i < len(self.fit_pct) else np.nan
            d[f"rmse_{sn}"]    = float(self.rmse[i])    if i < len(self.rmse)    else np.nan

        d.update(self.derivatives)
        d.update(self.aero)
        d.update(self.excitation)
        d.update(self.fisher_info)
        d.update(self.trim_kde)

        m = self.maneuver_2311
        for key in ("quality_2311", "score_2311_duration_mean", "score_2311_amplitude_mean"):
            d[key] = m.get(key, np.nan)
        for p in m.get("plateaus", []):
            pid = p.get("plateau", "?")
            d[f"2311_step{pid}_duration_s"]      = p.get("duration_s",      np.nan)
            d[f"2311_step{pid}_duration_score"]  = p.get("duration_score",  np.nan)
            d[f"2311_step{pid}_amplitude"]       = p.get("amplitude",       np.nan)
            d[f"2311_step{pid}_amplitude_score"] = p.get("amplitude_score", np.nan)
        for key in ("quality_phugoid", "actual_step_dur_s", "duration_score", "amplitude"):
            if key in m:
                d[f"phugoid_{key}"] = m[key]
        for key in ("quality_short_period", "actual_pulse_dur_s", "duration_score",
                    "linearity_score", "command_score", "max_bank_angle_deg",
                    "bank_angle_ok", "n_peaks"):
            if key in m:
                d[f"short_period_{key}"] = m[key]
        for pi, pk in enumerate(m.get("peaks", []), start=1):
            d[f"short_period_peak{pi}_rise_time_s"]          = pk.get("rise_time_s",          np.nan)
            d[f"short_period_peak{pi}_sharpness_score"]      = pk.get("sharpness_score",      np.nan)
            d[f"short_period_peak{pi}_clean_return_score"]   = pk.get("clean_return_score",   np.nan)
            d[f"short_period_peak{pi}_pretrim_duration_s"]   = pk.get("pretrim_duration_s",   np.nan)
            d[f"short_period_peak{pi}_trim_duration_score"]  = pk.get("trim_duration_score",  np.nan)
            d[f"short_period_peak{pi}_damping_duration_s"]   = pk.get("damping_duration_s",   np.nan)
            d[f"short_period_peak{pi}_damping_window_score"] = pk.get("damping_window_score", np.nan)

        return d

    def __repr__(self) -> str:
        q = self.maneuver_2311.get("quality_2311", float("nan"))
        return (
            f"Metrics(score={self.global_score:.3f}, dyn={self.dynamics}, "
            f"q2311={q:.3f})"
        )

# ─────────────────────────────────────────────────────────────────────────────
# ManeuverQualityAssessor
# ─────────────────────────────────────────────────────────────────────────────
class ManeuverQualityAssessor:
    DURATIONS_2311 = [2.0, 3.0, 1.0, 1.0]
    MIN_RUDDER_AFTER_AILERONS_S = 10.0

    def __init__(self, model: StateSpaceModel):
        self.model = model

    def generate_ideal_2311(
        self,
        data: "FlightData",
        trim: "TrimResult",
        elevator_col: Optional[str] = None,
        input_col: Optional[str] = None,
        first_step_window_s: float = 2.0,
        last_step_max_s: float = 1.0,
        soft_ramps: bool = False,
        ramp_tau_s: float = 0.15,
        score_col: Optional[str] = None,
        min_step_amplitude: float = 0.3,
        refine_window_s = 2.5,
        refine_onset_with_std: bool = True,
        hampel_window_s: float = 1.0,
    ) -> Tuple[np.ndarray, Dict]:
        col = input_col or elevator_col or data.config.col("elevator")

        elv = data.df[col].values.astype(float)
        dt  = data.dt

        window_n = max(1, int(round(hampel_window_s / dt)))
        elv = FlightData._hampel_filter(elv, window_n)
        N   = len(elv)
        t   = np.arange(N) * dt
        df_for_onset = pd.DataFrame({col: elv})

        i0_avg, i1_avg = int(trim.i0_avg), int(trim.i1_avg)
        trim_mean = float(np.nanmean(elv[i0_avg:i1_avg]))

        i_end = N

        scores_df = getattr(trim, "scores_df", None)
        seg = None
        if score_col is not None and scores_df is not None and score_col in scores_df.columns:
            score_vec   = pd.to_numeric(scores_df[score_col], errors="coerce").to_numpy(dtype=float)
            active_mask = np.isfinite(score_vec) & (score_vec < 0.95)
            search_floor = max(0, i1_avg)
            active_mask[:search_floor] = False
            min_len = max(3, int(round(0.5 / dt)))
            seg = TrimDetector._first_true_segment(active_mask, min_len=min_len)
        i_start_score = seg[0] if seg is not None else i1_avg

        if refine_onset_with_std:
            baseline_dur_s     = 8.0
            n_lookback         = int(round(refine_window_s / dt))
            n_baseline         = max(3, int(round(baseline_dur_s / dt)))
            baseline_i1_local  = max(0, i_start_score - n_lookback)
            baseline_i0_local  = max(0, baseline_i1_local - n_baseline)
            refine_n_sigma      = TrimDetector._refine_n_sigma_for_column(data.config, col)
            i_start = TrimDetector._refine_trim_end_control(
                df_for_onset, i_start_score, i_start_score, dt, col, n_sigma=refine_n_sigma,
                lookback_s=refine_window_s, extend_s=refine_window_s,
                baseline_i0=baseline_i0_local, baseline_i1=baseline_i1_local,
            )
        else:
            i_start = i_start_score

        if i_end <= i_start + 5:
            return data.U.copy(), {"status": "maneuver_too_short"}

        n_first    = int(round(first_step_window_s / dt))
        n_lookback = int(round(1.0 / dt))
        i_fs      = max(0, i_start - n_lookback)
        i_fe      = min(i_end, i_start + n_first)
        x_first   = elv[i_fs:i_fe]

        if len(x_first) < 5 or not np.any(np.isfinite(x_first)):
            return data.U.copy(), {"status": "first_step_too_short"}

        first_min_loc = int(np.nanargmin(x_first))
        first_max_loc = int(np.nanargmax(x_first))
        first_min     = float(x_first[first_min_loc])
        first_max     = float(x_first[first_max_loc])

        n_sustain = max(1, int(round(0.5 / dt)))
        onset_val = float(np.nanmean(elv[i_start:min(i_end, i_start + n_sustain)]))
        use_up    = onset_val >= trim_mean

        candidate_level = first_max if use_up else first_min
        if abs(candidate_level - trim_mean) < min_step_amplitude:
            use_up = not use_up

        if use_up:
            sequence_type     = "up-down"
            first_level       = first_max
            second_level      = float(np.nanmin(elv[i_start:i_end]))
            first_extreme_idx = i_fs + first_max_loc
        else:
            sequence_type     = "down-up"
            first_level       = first_min
            second_level      = float(np.nanmax(elv[i_start:i_end]))
            first_extreme_idx = i_fs + first_min_loc

        ideal_levels = [first_level, second_level, first_level, second_level]

        first_cross = self._find_next_crossing(
            elv, trim_mean, max(first_extreme_idx, i_start + 1), i_end
        )
        if first_cross is None:
            return data.U.copy(), {"status": "no_first_crossing", "sequence_type": sequence_type}

        y            = elv - trim_mean
        search_start = first_cross + max(2, int(0.10 / dt))
        min_sep      = max(3, int(0.25 / dt))
        raw_crossings: List[int] = []
        stop = i_end - 1
        if stop > search_start:
            y0_seg = y[search_start:stop]
            y1_seg = y[search_start + 1:stop + 1]
            valid  = np.isfinite(y0_seg) & np.isfinite(y1_seg)
            offsets = np.where(valid & ((y0_seg == 0) | (y0_seg * y1_seg < 0)))[0]
            k_idx = search_start + offsets
            raw_crossings = [
                int(k) for k in np.where(y0_seg[offsets] == 0, k_idx, k_idx + 1)
            ]

        next_crossings: List[int] = []
        for idx in raw_crossings:
            if not next_crossings or idx - next_crossings[-1] >= min_sep:
                next_crossings.append(int(idx))
        next_crossings = next_crossings[:2]

        base_bounds = [i_start, first_cross] + next_crossings
        while len(base_bounds) < 4:
            base_bounds.append(i_end)

        last_step_max_n = max(1, int(round(last_step_max_s / dt)))
        i_last_start    = int(base_bounds[3])
        i_last_end      = min(i_end, i_last_start + last_step_max_n)

        final_bounds = [int(b) for b in base_bounds[:4]] + [int(i_last_end)]

        elv_ideal             = np.full(N, np.nan, float)
        elv_ideal[:i_start]   = trim_mean
        elv_ideal[i_last_end:] = trim_mean

        plateau_infos: List[Dict] = []
        for j in range(4):
            i0, i1 = final_bounds[j], final_bounds[j + 1]
            if i1 <= i0:
                continue
            x_seg = elv[i0:i1]
            if len(x_seg) < 3 or not np.any(np.isfinite(x_seg)):
                continue
            if float(ideal_levels[j]) >= trim_mean:
                adj_val = float(np.nanmax(x_seg))
            else:
                adj_val = float(np.nanmin(x_seg))
            elv_ideal[i0:i1] = adj_val

            theo_dur  = self.DURATIONS_2311[j]
            real_dur  = float((i1 - i0) * dt)
            tol_base  = self.DURATIONS_2311[1] if j == 0 and len(self.DURATIONS_2311) > 1 else theo_dur
            dur_score = float(np.clip(1.0 - abs(real_dur - theo_dur) / max(tol_base, 0.1), 0.0, 1.0))
            plateau_infos.append({
                "plateau": j + 1,
                "i0": int(i0), "i1": int(i1),
                "t0": float(t[i0]), "t1": float(t[min(i1, N - 1)]),
                "duration_s": real_dur,
                "adjusted_value": adj_val,
                "amplitude": abs(adj_val - trim_mean),
                "theoretical_duration_s": theo_dur,
                "duration_score": dur_score,
            })

        if soft_ramps:
            n_ramp = max(3, int(round(ramp_tau_s / dt)))
            n_half = n_ramp // 2
            plat_vals = {p["plateau"] - 1: p["adjusted_value"] for p in plateau_infos}
            level_seq = [trim_mean] + [plat_vals.get(j, trim_mean) for j in range(4)]
            for j in range(len(final_bounds) - 1):
                b      = final_bounds[j]
                next_b = final_bounds[j + 1]
                y_from = level_seq[j]
                y_to   = level_seq[j + 1]
                if abs(y_to - y_from) < 1e-12:
                    continue
                win_start = b if j == 0 else max(i_start, b - n_half)
                win_end   = min(next_b, b + n_half)
                if win_end <= win_start:
                    continue
                alpha = np.linspace(0.0, 1.0, win_end - win_start)
                elv_ideal[win_start:win_end] = y_from + (y_to - y_from) * alpha
            i_le   = final_bounds[4]
            y_last = level_seq[4]
            win_e_s = max(i_start, i_le - n_half)
            win_e_e = min(N, i_le + n_half)
            if win_e_e > win_e_s and abs(trim_mean - y_last) > 1e-12:
                alpha = np.linspace(0.0, 1.0, win_e_e - win_e_s)
                elv_ideal[win_e_s:win_e_e] = y_last + (trim_mean - y_last) * alpha

        amps    = [p["amplitude"] for p in plateau_infos]
        amp_max = float(np.nanmax(amps)) if amps else np.nan
        for p in plateau_infos:
            p["amplitude_score"] = (
                float(np.clip(p["amplitude"] / amp_max, 0.0, 1.0))
                if np.isfinite(amp_max) and amp_max > 1e-12 else np.nan
            )

        dur_scores = [p["duration_score"] for p in plateau_infos]
        amp_scores = [p["amplitude_score"] for p in plateau_infos if np.isfinite(p["amplitude_score"])]
        dur_mean   = float(np.nanmean(dur_scores)) if dur_scores else np.nan
        amp_mean   = float(np.nanmean(amp_scores)) if amp_scores else np.nan
        q2311      = float(np.nanmean([dur_mean, amp_mean])) if (np.isfinite(dur_mean) and np.isfinite(amp_mean)) else np.nan

        elv_ideal_centered = elv_ideal - trim_mean

        U_ideal        = data.U - trim.U_mean
        U_ideal[:, 0]  = elv_ideal_centered

        debug = {
            "status":                "ok",
            "sequence_type":         sequence_type,
            "trim_mean":             trim_mean,
            "i_start":               int(i_start),
            "i_start_kde":           int(i1_avg),
            "i_end":                 int(i_end),
            "first_extreme_index":   int(first_extreme_idx),
            "first_extreme_time_s":  float(t[first_extreme_idx]),
            "first_crossing_index":  int(first_cross),
            "first_crossing_time_s": float(t[first_cross]),
            "crossings":             [int(first_cross)] + next_crossings,
            "plateaus":              plateau_infos,
            "score_2311_duration_mean":  dur_mean,
            "score_2311_amplitude_mean": amp_mean,
            "quality_2311":          q2311,
            "soft_ramps":            soft_ramps,
            "ramp_tau_s":            ramp_tau_s if soft_ramps else None,
        }
        return U_ideal, debug

    def generate_ideal_lateral_2311(self, data: "FlightData", trim: "TrimResult", **kwargs):
        cfg = data.config
        ail_diff_col = "_ail_diff"
        rud_col = cfg.col("rudder")
        U_id_rud, debug_rud = self.generate_ideal_2311(data, trim, elevator_col=rud_col, score_col="diff_rudder_1d", **kwargs, refine_window_s=2.5, refine_onset_with_std=True, hampel_window_s=0.1)
        U_id_ail, debug_ail = self.generate_ideal_2311(data, trim, elevator_col=ail_diff_col, score_col="diff_aileron1_1d", **kwargs, refine_window_s=0.5, refine_onset_with_std=True, hampel_window_s=0.1)

        ail_plateaus = debug_ail.get("plateaus") if debug_ail.get("status") == "ok" else None
        gap_s = None
        if ail_plateaus and debug_rud.get("status") == "ok":
            ail_end_s   = ail_plateaus[-1]["t1"]
            rud_start_s = debug_rud["i_start"] * data.dt
            gap_s = rud_start_s - ail_end_s
        rudder_timing_ok = gap_s is None or gap_s >= self.MIN_RUDDER_AFTER_AILERONS_S
        debug_rud["aileron_rudder_gap_s"] = gap_s
        debug_rud["rudder_timing_ok"]     = rudder_timing_ok

        ail_usable = debug_ail.get("status") == "ok"
        rud_usable = debug_rud.get("status") == "ok"

        U_ideal_combined = np.zeros_like(data.U)
        window = max(1, int(0.15 / data.dt))
        if ail_usable:
            U_ideal_combined[:, 0] = U_id_ail[:, 0]
        else:
            dev_ail = data.df[ail_diff_col].values - trim.U_mean[0]
            U_ideal_combined[:, 0] = pd.Series(dev_ail).rolling(window, center=True).mean().fillna(0).values
        if rud_usable:
            U_ideal_combined[:, 1] = U_id_rud[:, 0]
        else:
            dev_rud = data.df[rud_col].values - trim.U_mean[1]
            U_ideal_combined[:, 1] = pd.Series(dev_rud).rolling(window, center=True).mean().fillna(0).values
        global_status = "ok" if (ail_usable or rud_usable) else "error"
        qualities = []
        if ail_usable: qualities.append(debug_ail.get("quality_2311", np.nan))
        if rud_usable: qualities.append(debug_rud.get("quality_2311", np.nan))
        debug_combined = {
            "status": global_status,
            "quality_2311": float(np.nanmean(qualities)) if qualities else float("nan"),
            "aileron_rudder_gap_s": gap_s,
            "rudder_timing_ok": rudder_timing_ok,
            "debug_ail": debug_ail,
            "debug_rud": debug_rud
        }
        return U_ideal_combined, debug_combined

    @staticmethod
    def _adjust_peak_to_right_plateau(
        dev_raw: np.ndarray,
        peak_local: int,
        peak_dev: float,
        threshold_ratio: float = 0.95,
        max_search_s: float = 1.5,
        dt: float = 0.01,
    ) -> int:
        if not np.isfinite(peak_dev) or abs(peak_dev) < 1e-12:
            return peak_local

        max_search_n = int(max_search_s / dt)
        i_end = min(len(dev_raw) - 1, peak_local + max_search_n)

        threshold = threshold_ratio * abs(peak_dev)

        seg  = dev_raw[peak_local:i_end + 1]
        cond = np.isfinite(seg) & (np.sign(seg) == np.sign(peak_dev)) & (np.abs(seg) >= threshold)
        stop = np.where(~cond)[0]
        last_offset = int(stop[0]) - 1 if len(stop) else len(cond) - 1

        return peak_local + max(last_offset, 0)

    def generate_ideal_phugoid(
            self,
            data: "FlightData",
            trim: "TrimResult",
            elevator_col: Optional[str] = None,
            input_col: Optional[str] = None,
            ideal_step_dur_s: float = 3.0,
            smooth_window: int = 75,
            slope_threshold: float = 0.001,
            window_size: int = 25,
            min_segment_length: int = 100,
            guard_s: float = 0.5,
            min_amp: float = 0.05,
            trim_tol_ratio: float = 0.10,
            trim_tol_min: float = 0.02,
        ) -> Tuple[np.ndarray, Dict]:
        from scipy.optimize import least_squares

        def _fit_phugoid_shape_local(
                dev_raw,
                x_raw,
                trim_mean,
                peak_local,
                peak_dev,
                charge_start_local,
                return_to_trim_local,
                dt,
                peak_shift_s=5.0,
                gamma_bounds=(0.4, 3.5),
                max_nfev=80,
            ):
            sign_peak = np.sign(peak_dev)
            amp_peak = abs(peak_dev)

            max_shift_n = int(peak_shift_s / dt)
            i0 = max(charge_start_local + 2, peak_local - max_shift_n)
            i1 = min(return_to_trim_local - 2, peak_local + max_shift_n)

            if i1 <= i0:
                return {
                    "peak_local_fit": int(peak_local),
                    "charge_end_local": int(peak_local),
                    "return_to_trim_local": int(return_to_trim_local),
                    "gamma_charge": 1.0,
                    "gamma_return": 1.0,
                    "cost": float("nan"),
                    "success": False,
                }

            seg  = dev_raw[i0:i1 + 1]
            mask = np.isfinite(seg) & (np.sign(seg) == sign_peak) & (np.abs(seg) >= 0.70 * amp_peak)
            allowed = i0 + np.where(mask)[0]

            if len(allowed) == 0:
                allowed = np.arange(i0, i1 + 1)

            def build_candidate(peak_idx, gamma_charge, gamma_return):
                peak_idx = int(peak_idx)
                peak_idx = int(allowed[np.argmin(np.abs(allowed - peak_idx))])

                return_local = max(return_to_trim_local, peak_idx + 2)

                y = np.full(return_local - charge_start_local + 1, trim_mean, dtype=float)
                peak_val = float(x_raw[peak_idx])

                n_charge = peak_idx - charge_start_local + 1
                y[:n_charge] = _smoothstep_cos_shape(
                    trim_mean, peak_val, n_charge, gamma=gamma_charge,
                )

                n_return = return_local - peak_idx + 1
                y[n_charge - 1:] = _smoothstep_cos_shape(
                    peak_val, trim_mean, n_return, gamma=gamma_return,
                )

                return y, peak_idx, return_local

            _N_fixed = max(return_to_trim_local - charge_start_local + 1, 1)
            _y_real_fixed = x_raw[charge_start_local:charge_start_local + _N_fixed]
            _N_fixed = len(_y_real_fixed)

            def residual(p):
                peak_idx_cont, gamma_charge, gamma_return = p

                y_model, peak_idx, return_local = build_candidate(
                    peak_idx_cont, gamma_charge, gamma_return,
                )

                if len(y_model) >= _N_fixed:
                    y_m = y_model[:_N_fixed]
                else:
                    y_m = np.concatenate([
                        y_model,
                        np.full(_N_fixed - len(y_model), float(trim_mean)),
                    ])

                fin = np.isfinite(_y_real_fixed) & np.isfinite(y_m)
                if fin.sum() < 5:
                    return np.ones(_N_fixed + 2) * 1e3

                r_full = np.where(fin, y_m - _y_real_fixed, 0.0)

                peak_shift_penalty = 0.02 * ((peak_idx - peak_local) / max(max_shift_n, 1))
                amp_loss = max(0.0, 0.85 * amp_peak - abs(dev_raw[peak_idx]))
                amp_penalty = 0.05 * amp_loss / max(amp_peak, 1e-9)

                return np.r_[r_full, peak_shift_penalty, amp_penalty]

            x0 = np.array([float(peak_local), 1.0, 1.0])
            lower = np.array([float(allowed[0]),  gamma_bounds[0], gamma_bounds[0]])
            upper = np.array([float(allowed[-1]), gamma_bounds[1], gamma_bounds[1]])

            res = least_squares(
                residual,
                x0=x0,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=0.05,
                max_nfev=max_nfev,
            )

            y_best, peak_idx_best, return_best = build_candidate(
                res.x[0], res.x[1], res.x[2],
            )

            return {
                "peak_local_fit":      int(peak_idx_best),
                "charge_end_local":    int(peak_idx_best),
                "return_to_trim_local": int(return_best),
                "gamma_charge":        float(res.x[1]),
                "gamma_return":        float(res.x[2]),
                "cost":                float(res.cost),
                "success":             bool(res.success),
            }
        def _smoothstep_cos_shape(y0: float, y1: float, n: int, gamma: float = 1.0) -> np.ndarray:
            if n <= 1:
                return np.array([y1], dtype=float)

            s = np.linspace(0.0, 1.0, n)
            s = np.clip(s, 0.0, 1.0) ** gamma
            h = 0.5 - 0.5 * np.cos(np.pi * s)
            return y0 + (y1 - y0) * h

        col = input_col or elevator_col or data.config.col("elevator")
        elv = data.df[col].values.astype(float)
        dt = data.dt
        N = len(elv)
        t = np.arange(N) * dt

        i0_avg, i1_avg = int(trim.i0_avg), int(trim.i1_avg)
        trim_mean = float(np.nanmean(elv[i0_avg:i1_avg]))
        i_start = i1_avg

        if N <= i_start + 5:
            return data.U.copy(), {
                "status": "maneuver_too_short",
                "quality_phugoid": float("nan"),
            }

        x_maneuver = elv[i_start:N]

        if not np.any(np.isfinite(x_maneuver)):
            return data.U.copy(), {
                "status": "search_window_empty",
                "quality_phugoid": float("nan"),
            }

        if smooth_window > 1 and len(x_maneuver) > smooth_window:
            w = int(smooth_window)
            kernel = np.ones(w) / w
            padded = np.pad(x_maneuver, w // 2, mode="edge")
            x_smooth = np.convolve(padded, kernel, mode="valid")[:len(x_maneuver)]
        else:
            x_smooth = x_maneuver.copy()

        dev_smooth = x_smooth - trim_mean
        dev_raw = x_maneuver - trim_mean

        guard = min(max(int(guard_s / dt), 0), len(dev_raw) - 1)
        search_raw = dev_raw[guard:]

        if len(search_raw) < 5 or not np.any(np.isfinite(search_raw)):
            return data.U.copy(), {
                "status": "peak_search_window_empty",
                "quality_phugoid": float("nan"),
            }

        peak_local = guard + int(np.nanargmax(np.abs(search_raw)))
        peak_dev = float(dev_raw[peak_local])

        if not np.isfinite(peak_dev) or abs(peak_dev) < min_amp:
            return data.U.copy(), {
                "status": "phugoid_peak_too_small",
                "peak_dev_from_trim": peak_dev,
                "quality_phugoid": float("nan"),
            }

        direction = "down" if peak_dev < 0 else "up"
        tol_trim = max(trim_tol_min, trim_tol_ratio * abs(peak_dev))

        segments = []
        segments_summary = []

        if _HAS_TREND_ANALYZER:
            try:
                ta = _TrendAnalyzer(
                    slope_threshold=slope_threshold,
                    window_size=window_size,
                    min_segment_length=min_segment_length,
                )
                segments = ta.fit_single(x_smooth)
                segments_summary = [
                    (int(i_start + seg.start), int(i_start + seg.end), seg.trend)
                    for seg in segments
                ]
            except Exception:
                segments = []
                segments_summary = []

        charge_trend = "D" if direction == "down" else "U"

        charge_candidates = [
            s for s in segments
            if s.trend == charge_trend and s.start <= peak_local
        ]

        def _hybrid_score(seg):
            length = seg.end - seg.start + 1
            slope_score = abs(seg.slope)

            if seg.start <= peak_local <= seg.end:
                peak_factor = 1.0
            else:
                dist = min(abs(peak_local - seg.start), abs(peak_local - seg.end))
                peak_factor = np.exp(-3.0 * dist / max(len(x_smooth), 1))

            return (slope_score ** 1.0) * (length ** 0.5) * (peak_factor ** 2.0)

        if charge_candidates:
            charge_seg = max(charge_candidates, key=_hybrid_score)

            before_seg = np.where(np.abs(dev_smooth[:charge_seg.start]) <= tol_trim)[0]

            if before_seg.size > 0:
                charge_start_local = int(before_seg[-1])
            else:
                charge_start_local = int(charge_seg.start)
        else:
            before_peak = np.where(np.abs(dev_smooth[:peak_local]) <= tol_trim)[0]
            charge_start_local = (
                int(before_peak[-1]) if before_peak.size > 0
                else max(0, peak_local - int(2.0 / dt))
            )

        charge_end_local = self._adjust_peak_to_right_plateau(
            dev_raw=dev_raw,
            peak_local=int(peak_local),
            peak_dev=float(peak_dev),
            threshold_ratio=0.75,
            max_search_s=1.5,
            dt=dt,
        )

        opposite_trend = "D" if direction == "up" else "U"

        post_peak_segs = [
            s for s in segments
            if s.start > peak_local and s.trend == opposite_trend
        ]

        if post_peak_segs:
            discharge_seg = max(
                post_peak_segs,
                key=lambda s: abs(s.slope) * (s.end - s.start + 1),
            )
            return_to_trim_local = int(discharge_seg.end)

            dist_to_discharge = discharge_seg.start - peak_local
            if dist_to_discharge <= smooth_window:
                val_at_snap = float(x_maneuver[discharge_seg.start])
                peak_val_raw = float(x_maneuver[peak_local])
                rel_diff = (
                    abs(val_at_snap - peak_val_raw) / abs(peak_val_raw)
                    if abs(peak_val_raw) > 1e-9
                    else abs(val_at_snap - peak_val_raw)
                )
                if rel_diff <= 0.05:
                    charge_end_local = discharge_seg.start
        else:
            after_peak = dev_smooth[peak_local:]
            cands = (
                np.where(after_peak >= -tol_trim)[0] if peak_dev < 0
                else np.where(after_peak <= tol_trim)[0]
            )
            return_to_trim_local = (
                peak_local + int(cands[0]) if cands.size > 0
                else len(x_smooth) - 1
            )

        return_to_trim_local = max(return_to_trim_local, charge_end_local + 1)
        fit = _fit_phugoid_shape_local(
            dev_raw=dev_raw,
            x_raw=x_maneuver,
            trim_mean=trim_mean,
            peak_local=peak_local,
            peak_dev=peak_dev,
            charge_start_local=charge_start_local,
            return_to_trim_local=return_to_trim_local,
            dt=dt,
        )

        charge_end_local    = fit["charge_end_local"]
        return_to_trim_local = fit["return_to_trim_local"]
        best_gamma_charge   = fit["gamma_charge"]
        best_gamma_return   = fit["gamma_return"]
        charge_start_abs = min(max(i_start + charge_start_local, 0), N - 1)
        charge_end_abs = min(max(i_start + charge_end_local, 0), N - 1)
        return_to_trim_abs = min(max(i_start + return_to_trim_local, 0), N - 1)

        elv_ideal = np.full(N, trim_mean, float)
        peak_val = float(x_maneuver[charge_end_local])

        n_charge = charge_end_abs - charge_start_abs + 1
        if n_charge > 1:
            elv_ideal[charge_start_abs:charge_end_abs + 1] = _smoothstep_cos_shape(
                trim_mean, peak_val, n_charge, gamma=best_gamma_charge,
            )

        n_return = return_to_trim_abs - charge_end_abs + 1
        if n_return > 1:
            elv_ideal[charge_end_abs:return_to_trim_abs + 1] = _smoothstep_cos_shape(
                peak_val, trim_mean, n_return, gamma=best_gamma_return,
            )

        post_A, post_B, post_lam, post_omega, post_fit_ok = 0.0, 0.0, 0.05, 0.2, False
        post_omega_n  = float("nan")
        post_zeta     = float("nan")
        post_freq_hz  = float("nan")
        post_period_s = float("nan")
        post_i0 = return_to_trim_abs

        if post_i0 + 10 < N:
            t_post = np.arange(N - post_i0) * dt
            y_post_meas = elv[post_i0:N]
            fin_post = np.isfinite(y_post_meas)

            def _damped_sinusoid(t_arr, A, B, lam, omega):
                return trim_mean + np.exp(-lam * t_arr) * (
                    A * np.cos(omega * t_arr) + B * np.sin(omega * t_arr)
                )

            fitted = False

            if fin_post.sum() >= 10:
                y0 = float(y_post_meas[0] - trim_mean)

                amp_bound = max(abs(peak_dev) * 0.5, abs(y0), 0.05)

                x0_post = np.array([
                    y0,
                    0.0,
                    0.05,
                    0.2,
                ])

                lower_post = np.array([
                    -amp_bound,
                    -amp_bound,
                    1e-4,
                    0.05,
                ])

                upper_post = np.array([
                    amp_bound,
                    amp_bound,
                    2.0,
                    2.0,
                ])

                def _res_post(p):
                    A, B, lam, omega = p
                    y_model = _damped_sinusoid(t_post, A, B, lam, omega)

                    r = np.where(fin_post, y_model - y_post_meas, 0.0)

                    amp_penalty = 0.02 * max(
                        0.0,
                        np.sqrt(A**2 + B**2) - abs(peak_dev) * 0.3
                    )

                    return np.r_[r, amp_penalty]

                try:
                    rp = least_squares(
                        _res_post,
                        x0=x0_post,
                        bounds=(lower_post, upper_post),
                        loss="soft_l1",
                        f_scale=max(abs(peak_dev) * 0.03, 1e-3),
                        max_nfev=150,
                    )

                    post_A     = float(rp.x[0])
                    post_B     = float(rp.x[1])
                    post_lam   = float(rp.x[2])
                    post_omega = float(rp.x[3])
                    post_fit_ok = bool(rp.success)

                    _post_omega_n = float(np.sqrt(post_lam**2 + post_omega**2))
                    post_omega_n  = _post_omega_n
                    post_zeta     = float(post_lam / _post_omega_n) if _post_omega_n > 1e-12 else float("nan")
                    post_freq_hz  = float(post_omega / (2.0 * np.pi))
                    post_period_s = float(1.0 / post_freq_hz) if post_freq_hz > 1e-12 else float("nan")

                    y_junction = trim_mean + post_A

                    if n_return > 1:
                        elv_ideal[charge_end_abs:return_to_trim_abs + 1] = _smoothstep_cos_shape(
                            peak_val, y_junction, n_return, gamma=best_gamma_return,
                        )

                    t_post_full = np.arange(N - return_to_trim_abs) * dt
                    elv_ideal[return_to_trim_abs:N] = _damped_sinusoid(
                        t_post_full, post_A, post_B, post_lam, post_omega,
                    )
                    fitted = True

                except Exception as _exc_post:
                    print(f"    [WARN] post-maneuver sinusoid fit failed: {_exc_post}")
                    fitted = False

            if not fitted:
                elv_ideal[post_i0:] = trim_mean

        else:
            if post_i0 < N:
                elv_ideal[post_i0:] = trim_mean

        actual_step_dur = float((return_to_trim_abs - charge_start_abs) * dt)

        dur_score = float(np.clip(
            1.0 - abs(actual_step_dur - ideal_step_dur_s) / max(ideal_step_dur_s, 0.1),
            0.0,
            1.0,
        )) if ideal_step_dur_s > 0 else float("nan")

        seg_actual = elv[charge_start_abs:return_to_trim_abs + 1]
        seg_ideal = elv_ideal[charge_start_abs:return_to_trim_abs + 1]

        mask = np.isfinite(seg_actual) & np.isfinite(seg_ideal)

        if mask.sum() > 2:
            resid = seg_actual[mask] - seg_ideal[mask]
            tss = float(np.sum((seg_actual[mask] - np.mean(seg_actual[mask])) ** 2))
            rss = float(np.sum(resid ** 2))
            linearity_score = float(np.clip(1.0 - rss / max(tss, 1e-12), 0.0, 1.0))
        else:
            linearity_score = float("nan")

        finite_scores = [
            v for v in [dur_score, linearity_score]
            if np.isfinite(v)
        ]

        quality_phugoid = float(np.mean(finite_scores)) if finite_scores else float("nan")

        elv_ideal_centered = elv_ideal - trim_mean

        U_ideal = data.U - trim.U_mean
        U_ideal[:, 0] = elv_ideal_centered

        debug = {
            "status": "ok",
            "direction": direction,
            "trim_mean": trim_mean,
            "peak_val": peak_val,
            "peak_dev_from_trim": peak_dev,

            "i_start": int(i_start),
            "charge_start": int(charge_start_abs),
            "charge_end": int(charge_end_abs),
            "i_return_to_trim": int(return_to_trim_abs),
            "i_end": int(N),

            "charge_start_s": float(t[charge_start_abs]),
            "charge_end_s": float(t[charge_end_abs]),
            "return_to_trim_s": float(t[return_to_trim_abs]),

            "actual_step_dur_s": actual_step_dur,
            "ideal_step_dur_s": ideal_step_dur_s,

            "duration_score": dur_score,
            "linearity_score": linearity_score,
            "quality_phugoid": quality_phugoid,

            "smooth_window": int(smooth_window),
            "guard_s": float(guard_s),
            "trim_tol": float(tol_trim),
            "min_amp": float(min_amp),

            "segments_summary": segments_summary,
            "n_segments": len(segments),
            "fit_peak_local":  int(fit["peak_local_fit"]),
            "fit_peak_abs":    int(i_start + fit["peak_local_fit"]),
            "fit_gamma_charge": best_gamma_charge,
            "fit_gamma_return": best_gamma_return,
            "fit_cost":        fit["cost"],
            "fit_success":     fit["success"],

            "post_A": post_A,
            "post_B": post_B,
            "post_lam": post_lam,
            "post_omega": post_omega,
            "post_fit_ok": post_fit_ok,
            "post_omega_n":  post_omega_n,
            "post_zeta":     post_zeta,
            "post_freq_hz":  post_freq_hz,
            "post_period_s": post_period_s,

            "note": (
                "Phugoid ideal input: peak is detected on raw elevator signal and "
                "the ideal command is forced to pass through the extremum farthest "
                "from initial trim. Smoothed signal is used only for trend and "
                "return-to-trim detection."
            ),
        }

        return U_ideal, debug

    def generate_ideal_short_period(
        self,
        data,
        trim,
        elevator_col=None,
        input_col: Optional[str] = None,
        ideal_pulse_dur_s: float = 1.0,
        smooth_window: int = 15,
        guard_s: float = 0.1,
        min_amp: float = 0.02,
        trim_tol_ratio: float = 0.10,
        trim_tol_min: float = 0.01,
        return_tol_ratio: float = 0.05,
        return_tol_abs: float = 0.2,
        secondary_peak_amp_ratio: float = 0.60,
        secondary_peak_min_gap_s: float = 5.0,
        sharp_rise_cap_s: float = 2.0,
        max_pretrim_target_s: float = 10.0,
        max_damping_window_s: float = 10.0,
        max_bank_angle_deg_limit: float = 5.0,
    ) -> Tuple[np.ndarray, Dict]:
        from scipy.optimize import least_squares

        def _smoothstep_cos_shape(y0: float, y1: float, n: int, gamma: float = 1.0) -> np.ndarray:
            if n <= 1:
                return np.array([y1], dtype=float)
            s = np.linspace(0.0, 1.0, n)
            s = np.clip(s, 0.0, 1.0) ** gamma
            h = 0.5 - 0.5 * np.cos(np.pi * s)
            return y0 + (y1 - y0) * h

        col = input_col or elevator_col or data.config.col("elevator")
        elv = data.df[col].values.astype(float)
        dt       = data.dt
        N        = len(elv)
        t        = np.arange(N) * dt

        i0_avg, i1_avg = int(trim.i0_avg), int(trim.i1_avg)
        trim_mean = float(np.nanmean(elv[i0_avg:i1_avg]))
        i_start   = i1_avg

        if N <= i_start + 5:
            return data.U.copy(), {"status": "maneuver_too_short", "quality_short_period": float("nan")}

        x_maneuver = elv[i_start:N]
        if not np.any(np.isfinite(x_maneuver)):
            return data.U.copy(), {"status": "search_window_empty", "quality_short_period": float("nan")}

        dev_raw = x_maneuver - trim_mean
        tol_trim = max(abs(trim_tol_ratio * np.nanmax(np.abs(dev_raw))), trim_tol_min)

        guard      = min(max(int(guard_s / dt), 0), len(dev_raw) - 1)
        search_raw = dev_raw[guard:]

        if len(search_raw) < 5 or not np.any(np.isfinite(search_raw)):
            return data.U.copy(), {"status": "search_window_empty", "quality_short_period": float("nan")}

        peak_local = guard + int(np.nanargmax(np.abs(search_raw)))
        peak_dev   = float(dev_raw[peak_local])

        if not np.isfinite(peak_dev) or abs(peak_dev) < min_amp:
            return data.U.copy(), {"status": "peak_too_small", "quality_short_period": float("nan")}

        direction = "push" if peak_dev < 0 else "pull"

        from scipy.signal import find_peaks

        abs_dev       = np.abs(dev_raw)
        abs_dev_safe  = np.where(np.isfinite(abs_dev), abs_dev, -np.inf)
        local_max_idx, _ = find_peaks(abs_dev_safe, distance=max(1, int(0.2 / dt)))

        min_gap_samples     = max(int(np.ceil(secondary_peak_min_gap_s / dt)), 1)
        secondary_threshold = secondary_peak_amp_ratio * abs(peak_dev)

        candidate_idx = sorted(set(int(i) for i in local_max_idx if i >= guard) | {peak_local})
        qualifying = [i for i in candidate_idx if abs(dev_raw[i]) >= secondary_threshold]

        peak_locals: list[int] = []
        peak_devs:   list[float] = []
        for idx in qualifying:
            if peak_locals and idx - peak_locals[-1] < min_gap_samples:
                if abs(dev_raw[idx]) > abs(peak_devs[-1]):
                    peak_locals[-1] = idx
                    peak_devs[-1]   = float(dev_raw[idx])
                continue
            peak_locals.append(idx)
            peak_devs.append(float(dev_raw[idx]))

        n_peaks = len(peak_locals)

        charge_start_locals    = [0] * n_peaks
        return_to_trim_locals  = [0] * n_peaks
        prev_return_local      = -1

        for pi in range(n_peaks):
            p_local = peak_locals[pi]
            p_dev   = peak_devs[pi]
            next_peak_local = peak_locals[pi + 1] if pi + 1 < n_peaks else len(x_maneuver)

            lo_cs = prev_return_local + 1
            seg_cs = dev_raw[lo_cs:p_local]
            hits_cs = np.where(np.isfinite(seg_cs) & (np.abs(seg_cs) <= tol_trim))[0]
            cs_local = lo_cs + int(hits_cs[-1]) if len(hits_cs) else lo_cs
            charge_start_locals[pi] = cs_local

            return_threshold = max(return_tol_ratio * abs(p_dev), 0.0)
            search_end        = min(next_peak_local, len(x_maneuver))
            seg_rt = dev_raw[p_local:search_end]
            hits_rt = np.where(
                np.isfinite(seg_rt) & ((np.abs(seg_rt) <= return_threshold) | (np.abs(seg_rt) <= return_tol_abs))
            )[0]
            rt_local = p_local + int(hits_rt[0]) if len(hits_rt) else search_end - 1
            return_to_trim_locals[pi] = rt_local
            prev_return_local = rt_local

        charge_start_abs_list   = [min(max(i_start + c, 0), N - 1) for c in charge_start_locals]
        charge_end_abs_list     = [min(max(i_start + p, 0), N - 1) for p in peak_locals]
        return_to_trim_abs_list = [min(max(i_start + r, 0), N - 1) for r in return_to_trim_locals]
        peak_vals               = [float(x_maneuver[p]) for p in peak_locals]

        charge_start_abs   = charge_start_abs_list[0]
        charge_end_abs     = charge_end_abs_list[-1]
        return_to_trim_abs = return_to_trim_abs_list[-1]
        peak_val            = peak_vals[0]

        rise_time_s_list          = []
        sharpness_score_list      = []
        clean_return_score_list   = []
        pretrim_duration_s_list   = []
        trim_duration_score_list  = []
        damping_duration_s_list   = []
        damping_window_score_list = []

        for pi in range(n_peaks):
            p_local  = peak_locals[pi]
            cs_local = charge_start_locals[pi]
            rt_local = return_to_trim_locals[pi]

            rise_time_s = float((p_local - cs_local) * dt)
            sharpness_score = float(np.clip(1.0 - rise_time_s / max(sharp_rise_cap_s, 1e-9), 0.0, 1.0))

            seg = dev_raw[p_local:rt_local + 1]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 2:
                actual_path  = float(np.sum(np.abs(np.diff(seg))))
                minimal_path = float(abs(seg[0] - seg[-1]))
                clean_return_score = float(np.clip(minimal_path / max(actual_path, 1e-9), 0.0, 1.0))
            else:
                clean_return_score = float("nan")

            pretrim_start_abs = int(trim.i0) if pi == 0 else return_to_trim_abs_list[pi - 1]
            pretrim_duration_s = float((charge_start_abs_list[pi] - pretrim_start_abs) * dt)
            trim_duration_score = float(np.clip(pretrim_duration_s / max_pretrim_target_s, 0.0, 1.0))

            damping_duration_s = float((rt_local - p_local) * dt)
            if damping_duration_s <= max_damping_window_s:
                damping_window_score = 1.0
            else:
                damping_window_score = float(np.clip(
                    1.0 - (damping_duration_s - max_damping_window_s) / max_damping_window_s, 0.0, 1.0))

            rise_time_s_list.append(rise_time_s)
            sharpness_score_list.append(sharpness_score)
            clean_return_score_list.append(clean_return_score)
            pretrim_duration_s_list.append(pretrim_duration_s)
            trim_duration_score_list.append(trim_duration_score)
            damping_duration_s_list.append(damping_duration_s)
            damping_window_score_list.append(damping_window_score)

        phi_col = data.config.get_col("phi")
        if phi_col and phi_col in data.df.columns:
            phi = data.df[phi_col].values.astype(float)
            span = phi[charge_start_abs_list[0]: return_to_trim_abs_list[-1] + 1]
            span = span[np.isfinite(span)]
            if len(span) > 0:
                max_bank_angle_deg = float(np.degrees(np.nanmax(np.abs(span))))
                bank_angle_ok = bool(max_bank_angle_deg <= max_bank_angle_deg_limit)
            else:
                max_bank_angle_deg, bank_angle_ok = float("nan"), True
        else:
            max_bank_angle_deg, bank_angle_ok = float("nan"), True

        per_peak_scores = [s for group in (sharpness_score_list, clean_return_score_list,
                                            trim_duration_score_list, damping_window_score_list)
                           for s in group if np.isfinite(s)]
        command_score = (float(np.mean(per_peak_scores)) * (1.0 if bank_angle_ok else 0.0)
                         if per_peak_scores else float("nan"))

        elv_ideal = np.full(N, trim_mean, float)

        for pi in range(n_peaks):
            cs_abs, ce_abs, rt_abs = (
                charge_start_abs_list[pi], charge_end_abs_list[pi], return_to_trim_abs_list[pi],
            )
            pv = peak_vals[pi]

            n_charge = ce_abs - cs_abs + 1
            if n_charge > 1:
                elv_ideal[cs_abs:ce_abs + 1] = _smoothstep_cos_shape(
                    trim_mean, pv, n_charge,
                )

            n_return = rt_abs - ce_abs + 1
            if n_return > 1:
                elv_ideal[ce_abs:rt_abs + 1] = _smoothstep_cos_shape(
                    pv, trim_mean, n_return,
                )

        last_peak_val   = peak_vals[-1]
        last_peak_dev   = peak_devs[-1]
        last_n_return   = return_to_trim_abs - charge_end_abs + 1

        post_A, post_B, post_lam, post_omega, post_fit_ok = 0.0, 0.0, 0.5, 3.0, False
        post_i0 = return_to_trim_abs

        if post_i0 + 10 < N:
            t_post      = np.arange(N - post_i0) * dt
            y_post_meas = elv[post_i0:N]
            fin_post    = np.isfinite(y_post_meas)

            def _damped_sinusoid(t_arr, A, B, lam, omega):
                return trim_mean + np.exp(-lam * t_arr) * (
                    A * np.cos(omega * t_arr) + B * np.sin(omega * t_arr)
                )

            fitted = False
            if fin_post.sum() >= 10:
                y0        = float(y_post_meas[0] - trim_mean)
                amp_bound = max(abs(last_peak_dev) * 0.5, return_tol_abs, abs(y0), 0.02)

                x0_sp    = np.array([y0,  0.0, 0.5, 3.0])
                lower_sp = np.array([-amp_bound, -amp_bound, 0.05, 0.3])
                upper_sp = np.array([ amp_bound,  amp_bound, 15.0, 20.0])

                _N_fixed_sp   = len(t_post)
                _y_real_sp    = y_post_meas.copy()

                def _res_sp(p):
                    y_m = _damped_sinusoid(t_post, p[0], p[1], p[2], p[3])
                    r   = np.where(fin_post, y_m - _y_real_sp, 0.0)
                    amp_pen = 0.02 * max(0.0, np.sqrt(p[0]**2 + p[1]**2) - abs(last_peak_dev) * 0.3)
                    return np.r_[r, amp_pen]

                try:
                    rsp = least_squares(
                        _res_sp,
                        x0=x0_sp,
                        bounds=(lower_sp, upper_sp),
                        loss="soft_l1",
                        f_scale=max(abs(last_peak_dev) * 0.03, 1e-3),
                        max_nfev=150,
                    )
                    post_A     = float(rsp.x[0])
                    post_B     = float(rsp.x[1])
                    post_lam   = float(rsp.x[2])
                    post_omega = float(rsp.x[3])
                    post_fit_ok = bool(rsp.success)

                    y_junction = trim_mean + post_A
                    if last_n_return > 1:
                        elv_ideal[charge_end_abs:return_to_trim_abs + 1] = _smoothstep_cos_shape(
                            last_peak_val, y_junction, last_n_return,
                        )

                    t_post_full = np.arange(N - return_to_trim_abs) * dt
                    elv_ideal[return_to_trim_abs:N] = _damped_sinusoid(
                        t_post_full, post_A, post_B, post_lam, post_omega,
                    )
                    fitted = True

                except Exception as _exc_sp:
                    print(f"    [WARN] short-period post-pulse sinusoid fit failed: {_exc_sp}")

            if not fitted:
                elv_ideal[post_i0:] = trim_mean
        else:
            if post_i0 < N:
                elv_ideal[post_i0:] = trim_mean

        actual_pulse_dur = float((return_to_trim_abs - charge_start_abs) * dt)
        dur_score = float(np.clip(
            1.0 - abs(actual_pulse_dur - ideal_pulse_dur_s) / max(ideal_pulse_dur_s, 0.1),
            0.0, 1.0,
        )) if ideal_pulse_dur_s > 0 else float("nan")

        seg_actual = elv[charge_start_abs:return_to_trim_abs + 1]
        seg_ideal  = elv_ideal[charge_start_abs:return_to_trim_abs + 1]
        mask       = np.isfinite(seg_actual) & np.isfinite(seg_ideal)
        if mask.sum() > 2:
            resid = seg_actual[mask] - seg_ideal[mask]
            tss   = float(np.sum((seg_actual[mask] - np.mean(seg_actual[mask])) ** 2))
            rss   = float(np.sum(resid ** 2))
            linearity_score = float(np.clip(1.0 - rss / max(tss, 1e-12), 0.0, 1.0))
        else:
            linearity_score = float("nan")

        finite_scores   = [v for v in [dur_score, linearity_score] if np.isfinite(v)]
        quality_sp      = float(np.mean(finite_scores)) if finite_scores else float("nan")

        elv_ideal_centered = elv_ideal - trim_mean
        U_ideal            = data.U - trim.U_mean
        U_ideal[:, 0]      = elv_ideal_centered

        debug = {
            "status":           "ok",
            "direction":        direction,
            "trim_mean":        trim_mean,
            "peak_val":         peak_val,
            "peak_dev":         peak_dev,

            "i_start":          int(i_start),
            "charge_start":     int(charge_start_abs),
            "charge_end":       int(charge_end_abs),
            "i_return_to_trim": int(return_to_trim_abs),

            "charge_start_s":   float(t[charge_start_abs]),
            "charge_end_s":     float(t[charge_end_abs]),
            "return_to_trim_s": float(t[return_to_trim_abs]),

            "actual_pulse_dur_s": actual_pulse_dur,
            "ideal_pulse_dur_s":  ideal_pulse_dur_s,

            "duration_score":   dur_score,
            "linearity_score":  linearity_score,
            "quality_short_period": quality_sp,

            "post_A":       post_A,
            "post_B":       post_B,
            "post_lam":     post_lam,
            "post_omega":   post_omega,
            "post_fit_ok":  post_fit_ok,

            "n_peaks": n_peaks,
            "peaks": [
                {
                    "peak_val":         peak_vals[pi],
                    "peak_dev":         peak_devs[pi],
                    "charge_start_s":   float(t[charge_start_abs_list[pi]]),
                    "charge_end_s":     float(t[charge_end_abs_list[pi]]),
                    "return_to_trim_s": float(t[return_to_trim_abs_list[pi]]),

                    "rise_time_s":          rise_time_s_list[pi],
                    "sharpness_score":      sharpness_score_list[pi],
                    "clean_return_score":   clean_return_score_list[pi],
                    "pretrim_duration_s":   pretrim_duration_s_list[pi],
                    "trim_duration_score":  trim_duration_score_list[pi],
                    "damping_duration_s":   damping_duration_s_list[pi],
                    "damping_window_score": damping_window_score_list[pi],
                }
                for pi in range(n_peaks)
            ],

            "max_bank_angle_deg": max_bank_angle_deg,
            "bank_angle_ok":      bank_angle_ok,
            "command_score":      command_score,

            "segments_summary": [],
        }

        return U_ideal, debug

    def generate_ideal_dutch_roll(
        self,
        data: "FlightData",
        trim: "TrimResult",
        rudder_col: Optional[str] = None,
        input_col: Optional[str] = None,
        first_step_window_s: float = 2.0,
        ideal_doublet_dur_s: float = 2.0,
        guard_s: float = 0.1,
        min_amp: float = 0.02,
        trim_tol_ratio: float = 0.10,
        trim_tol_min: float = 0.01,
    ) -> Tuple[np.ndarray, Dict]:
        from scipy.optimize import least_squares

        def _smoothstep_cos_shape(y0: float, y1: float, n: int, gamma: float = 1.0) -> np.ndarray:
            if n <= 1:
                return np.array([y1], dtype=float)
            s = np.linspace(0.0, 1.0, n)
            s = np.clip(s, 0.0, 1.0) ** gamma
            h = 0.5 - 0.5 * np.cos(np.pi * s)
            return y0 + (y1 - y0) * h

        col = input_col or rudder_col or data.config.col("rudder")
        rud = data.df[col].values.astype(float)
        dt  = data.dt
        N   = len(rud)
        t   = np.arange(N) * dt

        i0_avg, i1_avg = int(trim.i0_avg), int(trim.i1_avg)
        trim_mean = float(np.nanmean(rud[i0_avg:i1_avg]))
        i_start   = i1_avg

        if N <= i_start + 5:
            return data.U.copy(), {"status": "maneuver_too_short", "quality_dutch_roll": float("nan")}

        x_maneuver = rud[i_start:N]
        if not np.any(np.isfinite(x_maneuver)):
            return data.U.copy(), {"status": "search_window_empty", "quality_dutch_roll": float("nan")}

        dev_raw  = x_maneuver - trim_mean
        tol_trim = max(abs(trim_tol_ratio * np.nanmax(np.abs(dev_raw))), trim_tol_min)

        onset_abs   = TrimDetector._refine_trim_end_control(
            data.df, i_start, N, dt, col, n_sigma=6.0, baseline_s=3.0,
        )
        onset_local = max(0, onset_abs - i_start)

        guard   = min(max(onset_local + int(guard_s / dt), 0), len(dev_raw) - 1)
        n_first = int(round(first_step_window_s / dt))
        i_fe1   = min(len(dev_raw), guard + n_first)
        search1 = dev_raw[guard:i_fe1]

        if len(search1) < 5 or not np.any(np.isfinite(search1)):
            return data.U.copy(), {"status": "search_window_empty", "quality_dutch_roll": float("nan")}

        peak1_local = guard + int(np.nanargmax(np.abs(search1)))
        peak1_dev   = float(dev_raw[peak1_local])

        if not np.isfinite(peak1_dev) or abs(peak1_dev) < min_amp:
            return data.U.copy(), {"status": "peak1_too_small", "quality_dutch_roll": float("nan")}

        seg_cs1 = dev_raw[0:peak1_local]
        hits_cs1 = np.where(np.isfinite(seg_cs1) & (np.abs(seg_cs1) <= tol_trim))[0]
        charge_start_local = int(hits_cs1[-1]) if len(hits_cs1) else 0

        cross1 = self._find_next_crossing(x_maneuver, trim_mean, peak1_local, len(x_maneuver))
        if cross1 is None:
            return data.U.copy(), {
                "status": "no_reversal_crossing", "quality_dutch_roll": float("nan"),
                "peak1_dev": peak1_dev,
            }

        i_fe2   = min(len(dev_raw), cross1 + n_first)
        search2 = dev_raw[cross1:i_fe2]
        if len(search2) < 5 or not np.any(np.isfinite(search2)):
            return data.U.copy(), {
                "status": "no_second_peak", "quality_dutch_roll": float("nan"),
                "peak1_dev": peak1_dev,
            }
        opp_side    = np.where(np.sign(search2) != np.sign(peak1_dev), search2, 0.0)
        peak2_local = cross1 + int(np.nanargmax(np.abs(opp_side)))
        peak2_dev   = float(dev_raw[peak2_local])

        if not np.isfinite(peak2_dev) or abs(peak2_dev) < min_amp:
            return data.U.copy(), {
                "status": "peak2_too_small", "quality_dutch_roll": float("nan"),
                "peak1_dev": peak1_dev,
            }

        cross2 = self._find_next_crossing(x_maneuver, trim_mean, peak2_local, len(x_maneuver))
        return_to_trim_local = cross2 if cross2 is not None else len(x_maneuver) - 1

        charge_start_abs   = min(max(i_start + charge_start_local,   0), N - 1)
        peak1_abs           = min(max(i_start + peak1_local,          0), N - 1)
        peak2_abs           = min(max(i_start + peak2_local,          0), N - 1)
        return_to_trim_abs = min(max(i_start + return_to_trim_local, 0), N - 1)

        amp             = max(abs(peak1_dev), abs(peak2_dev))
        peak1_val_ideal = trim_mean + np.sign(peak1_dev) * amp
        peak2_val_ideal = trim_mean + np.sign(peak2_dev) * amp

        rud_ideal = np.full(N, trim_mean, float)

        n_charge = peak1_abs - charge_start_abs + 1
        if n_charge > 1:
            rud_ideal[charge_start_abs:peak1_abs + 1] = _smoothstep_cos_shape(
                trim_mean, peak1_val_ideal, n_charge,
            )

        n_reversal = peak2_abs - peak1_abs + 1
        if n_reversal > 1:
            rud_ideal[peak1_abs:peak2_abs + 1] = _smoothstep_cos_shape(
                peak1_val_ideal, peak2_val_ideal, n_reversal,
            )

        n_return = return_to_trim_abs - peak2_abs + 1
        if n_return > 1:
            rud_ideal[peak2_abs:return_to_trim_abs + 1] = _smoothstep_cos_shape(
                peak2_val_ideal, trim_mean, n_return,
            )

        post_A, post_B, post_lam, post_omega, post_fit_ok = 0.0, 0.0, 0.3, 2.0, False
        post_i0 = return_to_trim_abs

        if post_i0 + 10 < N:
            t_post      = np.arange(N - post_i0) * dt
            y_post_meas = rud[post_i0:N]
            fin_post    = np.isfinite(y_post_meas)

            def _damped_sinusoid(t_arr, A, B, lam, omega):
                return trim_mean + np.exp(-lam * t_arr) * (
                    A * np.cos(omega * t_arr) + B * np.sin(omega * t_arr)
                )

            fitted = False
            if fin_post.sum() >= 10:
                y0        = float(y_post_meas[0] - trim_mean)
                amp_bound = max(amp * 0.5, 0.02)

                x0_dr    = np.array([y0,  0.0, 0.3, 2.0])
                lower_dr = np.array([-amp_bound, -amp_bound, 0.02, 0.2])
                upper_dr = np.array([ amp_bound,  amp_bound, 5.0, 15.0])

                _y_real_dr = y_post_meas.copy()

                def _res_dr(p):
                    y_m = _damped_sinusoid(t_post, p[0], p[1], p[2], p[3])
                    r   = np.where(fin_post, y_m - _y_real_dr, 0.0)
                    amp_pen = 0.02 * max(0.0, np.sqrt(p[0]**2 + p[1]**2) - amp * 0.3)
                    return np.r_[r, amp_pen]

                try:
                    rdr = least_squares(
                        _res_dr,
                        x0=x0_dr,
                        bounds=(lower_dr, upper_dr),
                        loss="soft_l1",
                        f_scale=max(amp * 0.03, 1e-3),
                        max_nfev=150,
                    )
                    post_A     = float(rdr.x[0])
                    post_B     = float(rdr.x[1])
                    post_lam   = float(rdr.x[2])
                    post_omega = float(rdr.x[3])
                    post_fit_ok = bool(rdr.success)

                    y_junction = trim_mean + post_A
                    if n_return > 1:
                        rud_ideal[peak2_abs:return_to_trim_abs + 1] = _smoothstep_cos_shape(
                            peak2_val_ideal, y_junction, n_return,
                        )

                    t_post_full = np.arange(N - return_to_trim_abs) * dt
                    rud_ideal[return_to_trim_abs:N] = _damped_sinusoid(
                        t_post_full, post_A, post_B, post_lam, post_omega,
                    )
                    fitted = True

                except Exception as _exc_dr:
                    print(f"    [WARN] dutch-roll post-doublet sinusoid fit failed: {_exc_dr}")

            if not fitted:
                rud_ideal[post_i0:] = trim_mean
        else:
            if post_i0 < N:
                rud_ideal[post_i0:] = trim_mean

        actual_doublet_dur = float((return_to_trim_abs - charge_start_abs) * dt)
        dur_score = float(np.clip(
            1.0 - abs(actual_doublet_dur - ideal_doublet_dur_s) / max(ideal_doublet_dur_s, 0.1),
            0.0, 1.0,
        )) if ideal_doublet_dur_s > 0 else float("nan")

        symmetry_score = float(np.clip(
            1.0 - abs(abs(peak1_dev) - abs(peak2_dev)) / max(amp, 1e-9), 0.0, 1.0,
        ))

        seg_actual = rud[charge_start_abs:return_to_trim_abs + 1]
        seg_ideal  = rud_ideal[charge_start_abs:return_to_trim_abs + 1]
        mask       = np.isfinite(seg_actual) & np.isfinite(seg_ideal)
        if mask.sum() > 2:
            resid = seg_actual[mask] - seg_ideal[mask]
            tss   = float(np.sum((seg_actual[mask] - np.mean(seg_actual[mask])) ** 2))
            rss   = float(np.sum(resid ** 2))
            linearity_score = float(np.clip(1.0 - rss / max(tss, 1e-12), 0.0, 1.0))
        else:
            linearity_score = float("nan")

        finite_scores  = [v for v in [symmetry_score, linearity_score] if np.isfinite(v)]
        quality_dr     = float(np.mean(finite_scores)) if finite_scores else float("nan")

        rud_ideal_centered = rud_ideal - trim_mean
        U_ideal             = data.U - trim.U_mean
        rudder_u_idx         = 1 if data.config.dynamics == "lateral" else 0
        U_ideal[:, rudder_u_idx] = rud_ideal_centered

        debug = {
            "status":           "ok",
            "trim_mean":        trim_mean,
            "peak1_val":        float(x_maneuver[peak1_local]),
            "peak1_dev":        peak1_dev,
            "peak2_val":        float(x_maneuver[peak2_local]),
            "peak2_dev":        peak2_dev,
            "symmetric_amplitude": amp,

            "i_start":          int(i_start),
            "charge_start":     int(charge_start_abs),
            "peak1_index":      int(peak1_abs),
            "peak2_index":      int(peak2_abs),
            "i_return_to_trim": int(return_to_trim_abs),

            "charge_start_s":   float(t[charge_start_abs]),
            "peak1_s":          float(t[peak1_abs]),
            "peak2_s":          float(t[peak2_abs]),
            "return_to_trim_s": float(t[return_to_trim_abs]),

            "actual_doublet_dur_s": actual_doublet_dur,
            "ideal_doublet_dur_s":  ideal_doublet_dur_s,

            "duration_score":   dur_score,
            "symmetry_score":   symmetry_score,
            "linearity_score":  linearity_score,
            "quality_dutch_roll": quality_dr,

            "post_A":       post_A,
            "post_B":       post_B,
            "post_lam":     post_lam,
            "post_omega":   post_omega,
            "post_fit_ok":  post_fit_ok,
        }

        return U_ideal, debug

    def compare(
        self,
        Xc: np.ndarray,
        Uc_real: np.ndarray,
        Uc_ideal: np.ndarray,
        dt: float,
        dynamics: str = "longitudinal",
        ideal_score_window: Optional[Tuple[int, int]] = None,
    ) -> Tuple["Metrics", "Metrics"]:
        n              = len(Uc_real)
        Uc_ideal_algnd = Uc_ideal[:n] if len(Uc_ideal) >= n else np.pad(
            Uc_ideal, ((0, n - len(Uc_ideal)), (0, 0)), mode="edge"
        )
        Xsim_real  = self.model.simulate_matching(Xc, Uc_real,        dt)
        Xsim_ideal = self.model.simulate_matching(Xc, Uc_ideal_algnd, dt)

        def _partial(Xsim, window=None):
            min_n = min(len(Xc), len(Xsim))
            Xc_s, Xsim_s = Xc[:min_n], Xsim[:min_n]
            if window is not None:
                s0, s1 = max(0, window[0]), min(min_n, window[1])
                if s1 > s0:
                    Xc_s, Xsim_s = Xc_s[s0:s1], Xsim_s[s0:s1]
            r2, pearson, fit_pct, rmse = Metrics._compute_reconstruction(Xc_s, Xsim_s)
            return Metrics(r2=r2, pearson=pearson, fit_pct=fit_pct, rmse=rmse, dynamics=dynamics)

        return _partial(Xsim_real), _partial(Xsim_ideal, ideal_score_window)

    @staticmethod
    def _find_next_crossing(x: np.ndarray, ref: float, start_idx: int, end_idx: int) -> Optional[int]:
        y = x - ref
        stop = min(end_idx, len(x) - 1)
        if stop <= start_idx:
            return None
        idx = np.arange(start_idx, stop)
        y0 = y[idx]
        y1 = y[idx + 1]
        valid = np.isfinite(y0) & np.isfinite(y1)
        hit = np.where(valid & ((y0 == 0) | (y0 * y1 < 0)))[0]
        if len(hit) == 0:
            return None
        offset = int(hit[0])
        k = int(idx[offset])
        return k if y0[offset] == 0 else k + 1

# ─────────────────────────────────────────────────────────────────────────────
# PipelineResult
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    file_path:     str
    data:          FlightData
    trim:          TrimResult
    model_moesp:   StateSpaceModel
    model:         StateSpaceModel
    Xc:            np.ndarray
    Uc:            np.ndarray
    Acc_c:         np.ndarray
    metrics:       Metrics
    maneuver_type: str = "2311"
    Xc_lowpass:    Optional[np.ndarray] = None
    Uc_lowpass:    Optional[np.ndarray] = None
    Acc_c_lowpass: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"PipelineResult({os.path.basename(self.file_path)}, "
            f"score={self.metrics.global_score:.3f}, "
            f"stable={self.model.is_stable})"
        )

# ─────────────────────────────────────────────────────────────────────────────
# IdentificationPipeline  — high-level façade  [VERSION AVEC TRACES]
# ─────────────────────────────────────────────────────────────────────────────
class IdentificationPipeline:
    _THRESHOLDS = [0.95, 0.90, 0.85, 0.80]

    def __init__(
        self,
        config: AircraftConfig,
        hyb=None,
        moesp_order: int = 4,
        SS_f: int = 30,
        lambda_x: float = 1.0,
        lambda_acc: float = 1.0,
        lambda_reg: float = 1.0,
        tv_weight: float = 20.0,
        aircraft_plane: int = 2,
        max_trim_avg_s: float = 30.0,
        downsample: bool = True,
        ds: int = 4,
        min_trim_dur_s: float = 3.0,
        maneuver_type: str = "2311",
        maneuver_name: str = "",
    ):
        self.config         = config
        self.aircraft_plane = aircraft_plane
        self.max_trim_avg_s = max_trim_avg_s
        self.min_trim_dur_s = float(min_trim_dur_s)
        self.maneuver_type  = maneuver_type
        self.maneuver_name  = maneuver_name
        self.data_kw        = dict(tv_weight=tv_weight)
        self.trimmer        = TrimDetector(hyb)
        self.moesp          = MOESPIdentifier(order=moesp_order, SS_f=SS_f)
        refiner_kw = dict(lambda_x=lambda_x, lambda_acc=lambda_acc,
                          lambda_reg=lambda_reg, downsample=downsample, ds=ds)
        if config.dynamics == "lateral":
            self.refiner = LateralGreyBoxRefinement(**refiner_kw)
        else:
            self.refiner = LongitudinalGreyBoxRefinement(**refiner_kw)

    def run(self, file_path: str, verbose: bool = False) -> PipelineResult:
        """Full pipeline with retry over trim detection parameters."""
        print(f"[TRACE] run(): chargement fichier {file_path} ...", flush=True)
        data = FlightData.from_file(file_path, self.config)
        print(f"[TRACE] run(): fichier chargé, N={len(data.df)} lignes", flush=True)
        data.preprocess(verbose=verbose, **self.data_kw)
        print(f"[TRACE] run(): preprocess terminé, N={len(data.df)}, durée={len(data.df)*data.dt:.1f}s", flush=True)

        score_priority = self._score_priority(
            dynamics=self.config.dynamics,
            maneuver_type=self.maneuver_type,
            maneuver_name=self.maneuver_name,
        )
        n_combos        = len(self._THRESHOLDS)
        best_result: Optional[PipelineResult] = None
        best_score      = -np.inf
        tested_windows: set = set()

        for i_try, threshold in enumerate(self._THRESHOLDS):
            is_last       = (i_try == n_combos - 1)
            avg_durations = self._avg_durations(i_try, is_last)

            for avg_dur_s in avg_durations:
                print(f"[TRACE] run(): tentative threshold={threshold} avg_dur={avg_dur_s}", flush=True)
                t0 = time.time()
                try:
                    trim = self.trimmer._detect_explicit(
                        data,
                        threshold=threshold,
                        score_col_priority=score_priority,
                        avg_duration_s=min(avg_dur_s, self.max_trim_avg_s),
                        is_last_try=is_last,
                        plane=self.aircraft_plane,
                        min_duration_s=self.min_trim_dur_s,
                        maneuver_name=self.maneuver_name,
                        refine_with_elevator=(self.maneuver_type == "frequency_sweep"),
                    )
                except RuntimeError as exc:
                    print(f"[TRACE] run(): _detect_explicit ECHEC en {time.time()-t0:.1f}s -> {exc}", flush=True)
                    continue
                print(f"[TRACE] run(): _detect_explicit OK en {time.time()-t0:.1f}s", flush=True)

                win_key = (trim.i0_avg, trim.i1_avg)
                if win_key in tested_windows:
                    print(f"[TRACE] run(): fenêtre {win_key} déjà testée, skip", flush=True)
                    continue
                tested_windows.add(win_key)

                print("[TRACE] run(): appel _fit_one (MOESP + NLS)...", flush=True)
                t0 = time.time()
                try:
                    result = self._fit_one(data, trim, file_path, verbose)
                except Exception as exc:
                    print(f"[TRACE] run(): _fit_one ECHEC en {time.time()-t0:.1f}s -> {exc}", flush=True)
                    if verbose:
                        print(f"  [retry] fit failed (thr={threshold:.2f}): {exc}")
                    continue
                print(f"[TRACE] run(): _fit_one OK en {time.time()-t0:.1f}s", flush=True)

                ok, reason = self._quality_ok(result)
                if ok:
                    print(f"[TRACE] run(): résultat accepté (thr={threshold}) -> retour", flush=True)
                    self._add_maneuver_metrics(result, self.maneuver_type, verbose)
                    return result

                print(f"[TRACE] run(): résultat rejeté -> {reason}", flush=True)
                if result.metrics.global_score > best_score:
                    best_score  = result.metrics.global_score
                    best_result = result
                if verbose:
                    print(
                        f"  [retry] thr={threshold:.2f} avg={avg_dur_s:.1f}s "
                        f"score={result.metrics.global_score:.3f} → {reason}"
                    )

        # All quality checks failed — return best attempt
        if best_result is not None:
            print("[TRACE] run(): tous les seuils testés, retour du meilleur résultat", flush=True)
            self._add_maneuver_metrics(best_result, self.maneuver_type, verbose)
            return best_result

        print("[TRACE] run(): AUCUN résultat exploitable -> RuntimeError", flush=True)
        raise RuntimeError(f"Identification échouée pour '{file_path}' à tous les seuils.")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fit_one(
        self,
        data: FlightData,
        trim: "TrimResult",
        file_path: str,
        verbose: bool,
    ) -> PipelineResult:
        """Run MOESP + grey-box refinement for one trim configuration."""
        use_lp = getattr(data, "use_accel_lowpass", False)

        Xc_full, Uc_full, Acc_c_full = data.center(trim.X_mean, trim.U_mean, trim.Acc_mean)
        if use_lp:
            data.use_accel_lowpass = False
            try:
                Xc_full_raw, Uc_full_raw, Acc_c_full_raw = data.center(
                    trim.X_mean, trim.U_mean, trim.Acc_mean
                )
            finally:
                data.use_accel_lowpass = True
        else:
            Xc_full_raw, Uc_full_raw, Acc_c_full_raw = Xc_full, Uc_full, Acc_c_full

        mask             = data.clean_mask(Xc_full, Uc_full)
        masked_positions = np.where(mask)[0]
        i0_in_masked     = int(np.searchsorted(masked_positions, trim.i0))

        Xc_all    = Xc_full[mask]
        Uc_all    = Uc_full[mask]
        Acc_c_all = Acc_c_full[mask]
        Xc_all_raw    = Xc_full_raw[mask]
        Uc_all_raw    = Uc_full_raw[mask]
        Acc_c_all_raw = Acc_c_full_raw[mask]

        Xc    = Xc_all[i0_in_masked:]
        Uc    = Uc_all[i0_in_masked:]
        Acc_c = Acc_c_all[i0_in_masked:]

        Xc_raw    = Xc_all_raw[i0_in_masked:]
        Uc_raw    = Uc_all_raw[i0_in_masked:]
        Acc_c_raw = Acc_c_all_raw[i0_in_masked:]

        dt = data.dt
        if data.config.dynamics == "lateral":
            sl     = slice(trim.i0_avg, trim.i1_avg)
            u0     = float(np.nanmean(data.df["_u"].values[sl]))
            w0     = float(np.nanmean(data.df["_w"].values[sl]))
            theta0 = float(np.nanmean(data.df[data.config.col("theta")].values[sl]))
            v0     = float(trim.X_mean[0])
        else:
            u0     = float(trim.X_mean[0])
            theta0 = float(trim.X_mean[3])
            w0     = float(trim.X_mean[1])
            v0     = 0.0

        print(f"[TRACE] _fit_one(): appel MOESP.fit  N={len(Xc)} ...", flush=True)
        t0 = time.time()
        model_moesp = self.moesp.fit(Xc, Uc, dt, u0=u0, theta0=theta0, w0=w0)
        print(f"[TRACE] _fit_one(): MOESP.fit terminé en {time.time()-t0:.1f}s", flush=True)

        print("[TRACE] _fit_one(): appel refiner.fit (NLS grey-box)...", flush=True)
        t0 = time.time()
        model = self.refiner.fit(model_moesp, Xc, Uc, Acc_c, dt)
        print(f"[TRACE] _fit_one(): refiner.fit terminé en {time.time()-t0:.1f}s", flush=True)

        print("[TRACE] _fit_one(): calcul metrics/aero/excitation/fisher...", flush=True)
        Xsim    = model.simulate_matching(Xc_raw, Uc_raw, dt)
        metrics = Metrics.build(Xc_raw, Xsim, model, data.config, trim, dt)
        metrics.aero = Metrics._compute_aero_metrics(
            Xc_raw, Xsim, data.config.dynamics, u0, w0, v0
        )
        metrics.excitation = Metrics._compute_excitation(Xc_raw, Uc_raw, data.config.dynamics)
        metrics.fisher_info = Metrics._compute_fisher_info(model, dynamics=data.config.dynamics)
        print("[TRACE] _fit_one(): metrics OK, fin de _fit_one", flush=True)

        if verbose:
            print(f"[Pipeline] {os.path.basename(file_path)}  →  {metrics}")

        return PipelineResult(
            file_path=file_path,
            data=data,
            trim=trim,
            model_moesp=model_moesp,
            model=model,
            Xc=Xc_raw,
            Uc=Uc_raw,
            Acc_c=Acc_c_raw,
            metrics=metrics,
            maneuver_type=self.maneuver_type,
            Xc_lowpass=(Xc if use_lp else None),
            Uc_lowpass=(Uc if use_lp else None),
            Acc_c_lowpass=(Acc_c if use_lp else None),
        )

    @staticmethod
    def _quality_ok(result: PipelineResult) -> Tuple[bool, str]:
        model = result.model

        ls_res = getattr(model, "_ls_result", None)
        if ls_res is not None:
            if ls_res.status == 0:
                return False, "NLS hit max_nfev (status=0)"
            if (ls_res.status in (1, 3)
                    and ls_res.nfev <= 5
                    and np.isfinite(ls_res.cost)
                    and ls_res.cost > 1e8):
                return False, f"convergence suspecte (status={ls_res.status}, nfev={ls_res.nfev}, cost={ls_res.cost:.3e})"

        rho = model.spectral_radius
        if rho > 1.02:
            return False, f"ρ={rho:.4f} > 1.02"

        Xsim = model.simulate_matching(result.Xc, result.Uc, result.data.dt)
        if not np.all(np.isfinite(Xsim)) or np.max(np.abs(Xsim)) > 1e6:
            return False, "simulation diverged"

        return True, ""

    @staticmethod
    def _add_maneuver_metrics(result: "PipelineResult", maneuver_type: str = "2311",
                              verbose: bool = False) -> None:
        try:
            assessor = ManeuverQualityAssessor(result.model)
            generator = getattr(assessor, f"generate_ideal_{maneuver_type}", None) or assessor.generate_ideal_2311
            _, debug = generator(result.data, result.trim)
            result.metrics.maneuver_2311 = debug
        except Exception as exc:
            if verbose:
                print(f"  [maneuver quality] skipped: {exc}")
            result.metrics.maneuver_2311 = {"status": f"error: {exc}"}

    def _score_priority(self, dynamics: str = "longitudinal", maneuver_type: str = "2311",
                         maneuver_name: str = "") -> Tuple[Union[str, Tuple[str, ...]], ...]:
        return self.trimmer._score_priority_for_plane(
            self.aircraft_plane,
            dynamics=dynamics,
            maneuver_type=maneuver_type,
            maneuver_name=maneuver_name,
        )

    def _avg_durations(self, i_try: int, is_last: bool) -> List[float]:
        if is_last:
            return [10.0, 8.0, 6.0, 4.0, 2.0]
        return [max(2.0, 30.0 - 2.0 * i_try)]


# ─────────────────────────────────────────────────────────────────────────────
# ElevatorActuatorCouplingIdentifier
# ─────────────────────────────────────────────────────────────────────────────
class ElevatorActuatorCouplingIdentifier:
    def __init__(self, col_col: Optional[str] = None, stick_free: bool = True):
        self.col_col    = col_col
        self.stick_free = stick_free

    def fit(self, result: "PipelineResult") -> Optional[dict]:
        data  = result.data
        trim  = result.trim
        dt    = data.dt

        eta_e = result.Uc[:, 0]

        w = result.Xc[:, 1]
        q = result.Xc[:, 2]

        col_col = self.col_col or data.config.get_col("column")
        if col_col is None or col_col not in data.df.columns:
            print(f"  [ElevActuator] colonne pilote introuvable ({col_col}), skip")
            return None

        eta_c_raw  = data.df[col_col].values
        eta_c_mean = float(eta_c_raw[trim.i0_avg:trim.i1_avg].mean())
        eta_c_c    = eta_c_raw - eta_c_mean

        Xc_full, Uc_full, _ = data.center(trim.X_mean, trim.U_mean, trim.Acc_mean)
        mask             = data.clean_mask(Xc_full, Uc_full)
        masked_positions = np.where(mask)[0]
        i0_in_masked     = int(np.searchsorted(masked_positions, trim.i0))
        eta_c_aligned    = eta_c_c[mask][i0_in_masked:]

        N = min(len(eta_e), len(eta_c_aligned), len(w), len(q))
        if N < 20:
            print(f"  [ElevActuator] segment trop court ({N} pts), skip")
            return None

        eta_e = eta_e[:N]
        eta_c = eta_c_aligned[:N]
        w     = w[:N]
        q     = q[:N]

        A_fixed = result.model.Ac
        B_fixed = result.model.Bc[:, 0]
        Xc_4    = result.Xc[:N]

        eta_e_dot = np.gradient(eta_e, dt)
        tau_init = float("nan")
        K_init   = float("nan")
        try:
            phi2   = np.column_stack([eta_e, eta_c])
            th2, _, _, _ = np.linalg.lstsq(phi2, eta_e_dot, rcond=None)
            at2, bc2 = float(th2[0]), float(th2[1])
            tau_init = float(-1.0 / at2) if abs(at2) > 1e-12 else float("nan")
            K_init   = float(bc2 * tau_init) if np.isfinite(tau_init) else float("nan")
        except Exception:
            pass

        _TAU_MIN, _TAU_MAX = 0.05, 8.0
        _K_LIM = 20.0

        tau0 = tau_init if (np.isfinite(tau_init) and _TAU_MIN <= tau_init <= _TAU_MAX) else 1.0
        K0   = K_init   if np.isfinite(K_init) else 1.0

        def _build_aug(tau_p, K_p, bw_p, bq_p):
            A_aug = np.zeros((5, 5))
            A_aug[:4, :4] = A_fixed
            A_aug[:4, 4]  = B_fixed
            A_aug[4, 1]   = bw_p
            A_aug[4, 2]   = bq_p
            A_aug[4, 4]   = -1.0 / tau_p
            B_aug = np.zeros((5, 1))
            B_aug[4, 0]   = K_p / tau_p
            return A_aug, B_aug

        def _residuals_coupled(p):
            tau_p, K_p, bw_p, bq_p = p
            A_aug, B_aug = _build_aug(tau_p, K_p, bw_p, bq_p)
            F_aug, G_aug = StateSpaceModel.zoh(A_aug, B_aug, dt)
            x_aug = np.zeros((N, 5))
            x_aug[0, :4] = Xc_4[0]
            x_aug[0, 4]  = eta_e[0]
            g_aug = G_aug[:, 0]
            for k in range(N - 1):
                x_aug[k + 1] = F_aug @ x_aug[k] + g_aug * eta_c[k]
            return (x_aug[:, :4] - Xc_4).ravel()

        if self.stick_free:
            def _fun(p): return _residuals_coupled([p[0], p[1], p[2], p[3]])
            x0_nls     = [tau0, K0, 0.0, 0.0]
            bounds_nls = ([_TAU_MIN, -_K_LIM, -np.inf, -np.inf],
                          [_TAU_MAX,  _K_LIM,  np.inf,  np.inf])
        else:
            def _fun(p): return _residuals_coupled([p[0], p[1], 0.0, 0.0])
            x0_nls     = [tau0, K0]
            bounds_nls = ([_TAU_MIN, -_K_LIM], [_TAU_MAX, _K_LIM])

        try:
            sol = least_squares(
                _fun, x0=x0_nls, bounds=bounds_nls,
                method="trf", ftol=1e-8, xtol=1e-8,
            )
            if self.stick_free:
                tau, K, bw, bq = [float(v) for v in sol.x]
            else:
                tau, K = [float(v) for v in sol.x]
                bw, bq = 0.0, 0.0
        except Exception as exc:
            print(f"  [ElevActuator] NLS failed: {exc}")
            return None

        a_tau = float(-1.0 / tau)
        b_c   = float(K / tau)

        A_aug_f, B_aug_f = _build_aug(tau, K, bw, bq)
        F_f, G_f = StateSpaceModel.zoh(A_aug_f, B_aug_f, dt)
        x_aug_f = np.zeros((N, 5))
        x_aug_f[0, :4] = Xc_4[0]
        x_aug_f[0, 4]  = eta_e[0]
        g_f = G_f[:, 0]
        for k in range(N - 1):
            x_aug_f[k + 1] = F_f @ x_aug_f[k] + g_f * eta_c[k]

        res_states = x_aug_f[:, :4] - Xc_4
        rmse   = float(np.sqrt(np.mean(res_states ** 2)))
        ss_tot = float(np.sum((Xc_4 - Xc_4.mean(axis=0)) ** 2))
        ss_res = float(np.sum(res_states ** 2))
        r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

        eta_e_dot_pred = a_tau * eta_e + b_c * eta_c + bw * w + bq * q
        var_pred   = float(np.var(eta_e_dot_pred))
        contrib_bw = float(np.var(bw * w)) / max(var_pred, 1e-12)
        contrib_bq = float(np.var(bq * q)) / max(var_pred, 1e-12)

        print(f"  [ElevActuator] init: tau_init={tau_init:.4f}s  K_init={K_init:.4f}")
        print(f"  [ElevActuator]  nls: tau={tau:.4f}s  K={K:.4f}  "
              f"bw={bw:.4f}  bq={bq:.4f}  R2={r2:.3f}  RMSE={rmse:.4f}  "
              f"contrib_bw={contrib_bw:.1%}  contrib_bq={contrib_bq:.1%}")

        return {
            "tau": tau, "K": K, "bw": bw, "bq": bq,
            "a_tau": a_tau, "b_c": b_c,
            "rmse": rmse, "r2": r2,
            "contrib_bw": contrib_bw, "contrib_bq": contrib_bq,
            "tau_init": tau_init, "K_init": K_init,
            "A_aug": A_aug_f,
            "B_aug": B_aug_f,
            "eta_e_sim": x_aug_f[:, 4],
        }
