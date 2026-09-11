"""
main.py
=======
Batch aircraft system identification — longitudinal 2311 maneuvers, un ou
plusieurs avions (cf. AIRCRAFTS ci-dessous). Équivalent à la cellule de
batch-processing de Moesp ABCD.ipynb, étendu pour boucler sur plusieurs avions
et fusionner les dossiers listés dans COMBINE_FOLDERS en un Excel unique.

Usage:
    python main.py

Outputs (par avion, dans RLS/Aircraft<N>/):
    results.xlsx              ← global (2 sheets : GLOBAL_RESULTS, ERRORS)
    results_2311_Column.xlsx  ← per-folder
    2311 Column/
        <maneuver_name>.pdf   ← one PDF per file

Outputs combinés (dans RLS/, un par dossier listé dans COMBINE_FOLDERS) :
    results_2311_Column_Aircraft2_Aircraft3.xlsx
"""

import os
import sys
import glob
import traceback
import warnings
from typing import Optional
from sklearn.exceptions import InconsistentVersionWarning

# Évite un crash UnicodeEncodeError sur les print() contenant des accents/emoji
# (✅, é, …) quand stdout n'est pas déjà UTF-8 — ex. sortie redirigée vers un
# fichier sous Windows (codepage cp1252 par défaut). Sans effet sur un
# terminal déjà UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import maneuver_quality_id
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import cloudpickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")            # headless rendering — must come before pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyFRF import FRF

from maneuver_quality_id import (
    AircraftConfig,
    IdentificationPipeline,
    ElevatorActuatorCouplingIdentifier,
    ManeuverQualityAssessor,
    Metrics,
    PipelineResult,
    StateSpaceModel,
)
# ─────────────────────────────────────────────────────────────────────────────
# Paths  (edit here to change aircraft / folders)
# ─────────────────────────────────────────────────────────────────────────────
RLS_ROOT = r"C:\Users\AW34220\Documents"
# (plane, dossier des données brutes, dossier de sortie, dossiers de manœuvre
# à traiter pour CET avion) — un passage par avion, dossiers de manœuvre
# indépendants d'un avion à l'autre (ex. Aircraft 1 teste des manœuvres
# différentes de 2/3, aucune manœuvre commune pour l'instant).
SHORT_PERIODS_FILTER_OUT = r"C:\Users\AW34220\Documents\Aircraft Short Periods Filter"
AIRCRAFTS: list[tuple[int, str, str, list[str], str]] = [
    # (3, r"C:\Users\irampon\Documents\Project MAYDAY\Aircrafts\AIRCRAFT 3\ETS_SHIP3",
    #     os.path.join(RLS_ROOT, "Aircraft3"),
    #     ["2311 Column"], "longitudinal"),
    # (2, r"C:\Users\irampon\Documents\Project MAYDAY\Aircrafts\AIRCRAFT 2\Reorganized Datas Pickle",
    #     os.path.join(RLS_ROOT, "Aircraft2"),
    #     ["2311 Column"], "longitudinal"),
    # Sortie séparée (Aircraft2_Lateral, pas Aircraft2) : _run_for_aircraft écrit un
    # results.xlsx global qui écrase tout à chaque appel -- avec le même out_dir que
    # l'entrée "2311 Column" ci-dessus, ce second appel (dynamics différente => pipeline
    # différent, donc forcément un appel séparé) effacerait ses résultats.
    (2, r"C:\Users\AW34220\Documents\Aircrafts\AIRCRAFT 2\Reorganized Datas Pickle",
        os.path.join(RLS_ROOT, "Aircraft2_Lateral"),
        ["2311 WheelPedal"], "lateral"),
]
# Dossiers (noms canoniques) dont les lignes sont, en plus des sorties par
# avion habituelles, fusionnées en un seul Excel multi-avions dans RLS_ROOT
# (ex. results_combined.xlsx, une feuille par dossier) — seuls les avions
# qui traitent effectivement ce dossier (cf. AIRCRAFTS ci-dessus) y
# contribuent ; un dossier propre à un seul avion n'est simplement fusionné
# qu'avec lui-même (aucun effet, pas d'erreur).
PLANE = 2
COMBINE_FOLDERS = [
    "2311 Column",
]

# Certains dossiers de manœuvre n'ont pas le même nom d'un avion à l'autre
# (ex. Aircraft 2 : "2311 WheelPedal", Aircraft 3 : "2311 Wheel_Pedal"). Cette
# table associe le nom canonique (utilisé dans AIRCRAFTS/COMBINE_FOLDERS, dans
# les exports "Folder", et pour la fusion cross-avions) au nom réel du dossier
# sur disque, par avion, quand il diffère. Un avion sans entrée ici pour un
# dossier donné utilise le nom canonique tel quel.
FOLDER_NAME_OVERRIDES: dict[str, dict[int, str]] = {
    # inutile pour les Short Periods (noms identiques sur disque)
}


def _resolve_folder_name(canonical_name: str, plane: int) -> str:
    return FOLDER_NAME_OVERRIDES.get(canonical_name, {}).get(plane, canonical_name)


HYB_DIR      = r"C:\Users\AW34220\Documents\best_models"   # KDE classifier dir

DYNAMICS         = "longitudinal"  # "longitudinal" or "lateral" — fallback pour les appelants hors AIRCRAFTS
                                     # (AIRCRAFTS porte désormais sa propre dynamics par entrée)
FS_DEFAULT       = 100.0           # fallback sampling frequency (Hz)  ← same as notebook
SS_F             = 20              # MOESP block rows
EXPORT_PDF              = True
EXPORT_AIAA_KDE         = False   # save AIAA KDE score subfigures (3×1 in, one per score col)
EXPORT_AIAA_COLORED     = False   # save AIAA signal subfigures colored by combined KDE score
EXPORT_FRF              = True    # ajoute une page FRF MIMO (pyFRF) vs modèle grey-box
USE_ACTUATOR_PIPELINE   = False     # utilise IdentificationPipelineActuator pour "2311 Column"
USE_ACCEL_LOWPASS_RECON = False     # reconstruction (Acc_c → grey-box NLS) sur ax/ay/az passe-bas
                                     # au lieu du signal brut (cf. FlightData.preprocess)
ACCEL_LOWPASS_FC_HZ     = 0.5      # fréquence de coupure (Hz) du passe-bas ax/ay/az
DOWNSAMPLE       = True            # downsample les données avant le grey-box NLS
DS               = 8               # facteur de downsample NLS (2, 4, 8…)
STICK_FREE       = False
SOFT_RAMPS_2311  = False           # True = pentes douces (premier ordre) aux transitions 2311
RAMP_TAU_S       = 0.5            # constante de temps de la pente (secondes)
# durée minimale du segment de trim par type de manœuvre
# clé = sous-chaîne (insensible à la casse) présente dans le nom du dossier
_MIN_TRIM_DUR_RULES: list[tuple[str, float]] = [
    ("phugoid",      3.0),   # phugoids lents
    ("short_period", 2.0),   # court période — trim plus court
    ("short period", 2.0),
    ("2311",         3.0),   # manœuvres 2311
]
_MIN_TRIM_DUR_DEFAULT = 3.0          # tous les autres dossiers


def _min_trim_dur(folder_name: str) -> float:
    fl = folder_name.lower()
    for keyword, dur in _MIN_TRIM_DUR_RULES:
        if keyword in fl:
            return dur
    return _MIN_TRIM_DUR_DEFAULT


# type de manœuvre par dossier (détermine le générateur idéal utilisé)
_MANEUVER_TYPE_RULES: list[tuple[str, str]] = [
    ("phugoid",          "phugoid"),
    ("short_period",     "short_period"),
    ("short period",     "short_period"),
    ("dutch roll",       "dutch_roll"),
    ("frequency sweep",  "frequency_sweep"),
    ("column sweep",     "column_sweep"),
    ("2311",             "2311"),
]
_MANEUVER_TYPE_DEFAULT = "unclassified"   # ne doit PAS être "2311" : un dossier non
    # reconnu doit rester réellement non reconnu (aucun generate_ideal_unclassified),
    # sinon le dispatch générique par nom (getattr(assessor, f"generate_ideal_{mtype}"))
    # le confondrait à tort avec un vrai doublet 2311 et produirait une analyse erronée.


def _maneuver_type(folder_name: str) -> str:
    fl = folder_name.lower()
    for keyword, mtype in _MANEUVER_TYPE_RULES:
        if keyword in fl:
            return mtype
    return _MANEUVER_TYPE_DEFAULT


# Set to a full file path to process only that one file (None = process all)
SINGLE_FILE = None

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
_STATE_LABELS = {
    "longitudinal": ["u (m/s)", "w (m/s)", "q (rad/s)", "θ (rad)"],
    "lateral":      ["v (m/s)", "p (rad/s)", "r (rad/s)", "φ (rad)"],
}


def _excitation_str(m: Metrics) -> str:
    """
    Formatte le diagnostic de persistance d'excitation (Metrics.excitation :
    λ_min/λ_max/κ de R_φ = (1/N)Σ φ_kφ_k^T, φ_k=[Xc[k];Uc[k]] normalisé, plus
    l'amplitude crête-à-crête mesurée de chaque canal état/commande) pour
    affichage dans le titre des figures de reconstruction. κ grand = au moins
    deux régresseurs colinéaires (mal dissociés) ; λ_min proche de 0 = au
    moins une direction quasi non excitée (paramètres associés peu fiables),
    indépendamment du score de reconstruction R²/Pearson/Fit%. Les amplitudes
    permettent de vérifier si un λ_min/κ faible s'explique simplement par une
    petite commande appliquée (attendu) ou par une vraie sous-excitation.
    """
    exc = m.excitation or {}
    if "excitation_kappa" not in exc:
        return "Excitation persistante : n/a"
    kappa = exc["excitation_kappa"]
    kappa_str = "∞" if not np.isfinite(kappa) else f"{kappa:.1f}"
    flag = "  [!] rang déficient" if exc.get("excitation_rank_deficient") else ""
    line1 = (
        f"Excitation persistante : λ_min(R_φ)={exc['excitation_lambda_min']:.4f}  "
        f"λ_max={exc['excitation_lambda_max']:.4f}  κ(R_φ)={kappa_str}  "
        f"(n_régresseurs={exc.get('excitation_n_regressors', '?')}){flag}"
    )
    amp_x = [(k[len("excitation_amp_x_"):], v) for k, v in exc.items()
             if k.startswith("excitation_amp_x_")]
    amp_u = [(k[len("excitation_amp_u_"):], v) for k, v in exc.items()
             if k.startswith("excitation_amp_u_")]
    parts = []
    if amp_x:
        parts.append("états=" + " ".join(f"{n}:{v:.3f}" for n, v in amp_x))
    if amp_u:
        parts.append("commandes=" + " ".join(f"{n}:{v:.4f}" for n, v in amp_u))
    if not parts:
        return line1
    return line1 + "\nAmplitude crête-à-crête —  " + "   ".join(parts)


def _full_X_raw_and_lowpass(data) -> tuple:
    """(X_full, X_full_lowpass) — X_full est TOUJOURS le signal réel (bascule
    data.use_accel_lowpass à False le temps de l'appel si besoin) ;
    X_full_lowpass est la version passe-bas si le flag était actif, sinon
    None. Utilisé pour que les figures affichent le signal réel comme
    référence, avec le passe-bas en superposition optionnelle."""
    was_lp = getattr(data, "use_accel_lowpass", False)
    if not was_lp:
        return data.X, None
    data.use_accel_lowpass = False
    try:
        X_full = data.X
    finally:
        data.use_accel_lowpass = True
    X_full_lp = data.X
    return X_full, X_full_lp


def _fig_reconstruction_standard(result: PipelineResult, title: str) -> plt.Figure:
    """
    6-panel figure (longitudinal: 4 states + TAS + alpha) or
    5-panel figure (lateral: 4 states + beta).
    Green span = trim segment, orange span = averaging window, red line = model start.
    """
    data  = result.data
    trim  = result.trim
    dt    = data.dt
    m     = result.metrics
    dyn   = data.config.dynamics

    # Full uncentered measured signal (toujours réel — cf. _full_X_raw_and_lowpass)
    X_full, X_full_lp = _full_X_raw_and_lowpass(data)   # (N_full, 4) [, (N_full, 4)]
    t_full = np.arange(len(X_full)) * dt

    # Simulated signal (centred) → uncentred for display
    Xsim_c = result.model.simulate_matching(result.Xc, result.Uc, dt)
    Xsim   = Xsim_c + trim.X_mean               # uncentre
    t_sim  = np.arange(trim.i0, trim.i0 + len(Xsim)) * dt

    # ── Aero reconstruction (over trim window, measured and simulated) ────────
    N_s = len(Xsim_c)
    if dyn == "lateral":
        v0 = float(trim.X_mean[0])
        sl = slice(int(trim.i0_avg), int(trim.i1_avg))
        u0 = float(np.nanmean(data.df["_u"].values[sl]))
        w0 = float(np.nanmean(data.df["_w"].values[sl]))
    else:
        u0, w0, v0 = float(trim.X_mean[0]), float(trim.X_mean[1]), 0.0

    TAS_m, alpha_m, beta_m, TAS_s, alpha_s, beta_s = Metrics._aero_signals(
        result.Xc[:N_s], Xsim_c, dyn, u0, w0, v0
    )
    alpha_m, alpha_s = np.degrees(alpha_m), np.degrees(alpha_s)
    beta_m,  beta_s  = np.degrees(beta_m),  np.degrees(beta_s)

    if dyn == "lateral":
        aero_panels = [(beta_m,  beta_s,  "β (deg)",    "beta")]
    else:
        aero_panels = [(TAS_m,   TAS_s,   "TAS (m/s)",  "tas"),
                       (alpha_m, alpha_s, "α (deg)",    "alpha")]

    n_panels = 4 + len(aero_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 2.5 * n_panels + 1), sharex=True)
    fig.suptitle(
        f"{title}\n"
        f"Global score = {m.global_score:.3f}  |  "
        f"stable = {result.model.is_stable}  |  "
        f"ρ = {result.model.spectral_radius:.4f}  |  "
        f"trim [{trim.i0*dt:.1f}s – {trim.i1*dt:.1f}s]  "
        f"avg [{trim.i0_avg*dt:.1f}s – {trim.i1_avg*dt:.1f}s]\n"
        f"{_excitation_str(m)}",
        fontsize=9,
    )

    def _span_and_vline(ax, i):
        ax.axvspan(trim.i0 * dt, trim.i1 * dt,
                   color="limegreen", alpha=0.12,
                   label="Trim" if i == 0 else None)
        ax.axvspan(trim.i0_avg * dt, trim.i1_avg * dt,
                   color="orange", alpha=0.18,
                   label="Avg window" if i == 0 else None)
        ax.axvline(trim.i0 * dt, color="red", ls=":", lw=1.2, alpha=0.8,
                   label="Model start" if i == 0 else None)

    # ── State panels (rows 0-3) ───────────────────────────────────────────────
    for i, (ax, lbl) in enumerate(zip(axes[:4], _STATE_LABELS[dyn])):
        ax.plot(t_full, X_full[:, i], "k-",  lw=0.8, alpha=0.55, label="Measured")
        if X_full_lp is not None:
            ax.plot(t_full, X_full_lp[:, i], color="green", lw=0.9, alpha=0.85,
                    label="Measured (passe-bas)" if i == 0 else None)
        ax.plot(t_sim,  Xsim[:, i],   "b--", lw=1.4,              label="Simulated")
        _span_and_vline(ax, i)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.01, 0.96,
            f"R²={m.r2[i]:.3f}  Pearson={m.pearson[i]:.3f}  Fit%={m.fit_pct[i]:.3f}",
            transform=ax.transAxes, fontsize=8, color="navy", va="top",
        )

    axes[0].legend(fontsize=8, loc="upper right", ncol=2)

    # ── Aero panels (rows 4-6) ────────────────────────────────────────────────
    for j, (ym, ys, lbl, key) in enumerate(aero_panels):
        ax = axes[4 + j]
        ax.plot(t_sim, ym, "k-",  lw=0.8, alpha=0.55, label="Measured")
        ax.plot(t_sim, ys, "b--", lw=1.4,              label="Simulated")
        _span_and_vline(ax, 99)   # 99 → no legend labels duplicated
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, alpha=0.3)
        r2_v      = m.aero.get(f"{key}_r2",      float("nan"))
        pearson_v = m.aero.get(f"{key}_pearson", float("nan"))
        fit_v     = m.aero.get(f"{key}_fit_pct", float("nan"))
        ax.text(
            0.01, 0.96,
            f"R²={r2_v:.3f}  Pearson={pearson_v:.3f}  Fit%={fit_v:.3f}",
            transform=ax.transAxes, fontsize=8, color="darkgreen", va="top",
        )

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    return fig


def _get_coupling(result: PipelineResult) -> Optional[dict]:
    """Extrait le dict coupling depuis result.metrics.derivatives, ou None."""
    d = result.metrics.derivatives
    if "tau_act" not in d:
        return None
    c = {k: d.get(k, float("nan")) for k in
         ("tau_act", "K_act", "bw", "bq", "r2_act", "rmse_act",
          "a_tau_act", "b_c_act", "cbw_act", "cbq_act")}
    c["eta_e_sim"] = getattr(result.metrics, "_eta_e_sim", None)
    return c


def _fig_reconstruction_with_coupling(result: PipelineResult, title: str,
                                       coupling: dict) -> plt.Figure:
    """
    Figure de reconstruction avec panneaux actionneur OLS.
    Panels : 4 états aéro | ηc + ηe mesurée | ηe mesurée vs reconstruite (OLS) |
             TAS | α | texte matrices A, B + paramètres actionneur.
    """
    data  = result.data
    trim  = result.trim
    dt    = data.dt
    m     = result.metrics
    dyn   = data.config.dynamics

    tau   = coupling.get("tau_act",  float("nan"))
    K     = coupling.get("K_act",    float("nan"))
    bw    = coupling.get("bw",       float("nan"))
    bq    = coupling.get("bq",       float("nan"))
    r2    = coupling.get("r2_act",   float("nan"))
    rmse  = coupling.get("rmse_act", float("nan"))
    a_tau = coupling.get("a_tau_act",float("nan"))
    b_c   = coupling.get("b_c_act",  float("nan"))
    cbw   = coupling.get("cbw_act",  float("nan"))
    cbq   = coupling.get("cbq_act",  float("nan"))

    # ── Signal de base (toujours réel — cf. _full_X_raw_and_lowpass) ──────────
    X_full, X_full_lp = _full_X_raw_and_lowpass(data)
    t_full = np.arange(len(X_full)) * dt

    Xsim_c = result.model.simulate_matching(result.Xc, result.Uc, dt)
    Xsim   = Xsim_c + trim.X_mean
    N_s    = len(Xsim_c)
    t_sim  = np.arange(trim.i0, trim.i0 + N_s) * dt

    u0, w0 = float(trim.X_mean[0]), float(trim.X_mean[1])
    TAS_m, alpha_m, _, TAS_s, alpha_s, _ = Metrics._aero_signals(
        result.Xc[:N_s], Xsim_c, dyn, u0, w0, 0.0
    )
    alpha_m, alpha_s = np.degrees(alpha_m), np.degrees(alpha_s)

    # ── ηe et ηc ─────────────────────────────────────────────────────────────
    eta_e = result.Uc[:N_s, 0]
    w_c   = result.Xc[:N_s, 1]
    q_c   = result.Xc[:N_s, 2]

    col_col   = data.config.get_col("column")
    eta_c_arr = np.zeros(N_s)
    if col_col and col_col in data.df.columns:
        eta_c_raw  = data.df[col_col].values
        eta_c_mean = float(eta_c_raw[trim.i0_avg:trim.i1_avg].mean())
        eta_c_c    = eta_c_raw - eta_c_mean
        Xc_f, Uc_f, _ = data.center(trim.X_mean, trim.U_mean, trim.Acc_mean)
        mask_c    = data.clean_mask(Xc_f, Uc_f)
        mpos      = np.where(mask_c)[0]
        i0m       = int(np.searchsorted(mpos, trim.i0))
        eta_c_all = eta_c_c[mask_c][i0m:]
        eta_c_arr = eta_c_all[:N_s]

    # ── ηe reconstruit : simulation couplée si disponible, sinon ZOH ─────────
    eta_e_sim_arr = coupling.get("eta_e_sim")
    if eta_e_sim_arr is not None and len(eta_e_sim_arr) >= N_s:
        eta_e_rec = np.asarray(eta_e_sim_arr)[:N_s]
        rec_label = "ηe reconstruit (sim. couplée)"
    else:
        eta_e_rec = np.zeros(N_s)
        if np.isfinite(a_tau) and np.isfinite(b_c) and np.isfinite(bw) and np.isfinite(bq):
            phi_dt  = float(np.exp(a_tau * dt))
            gam_dt  = (phi_dt - 1.0) / a_tau if abs(a_tau) > 1e-12 else dt
            u_act   = b_c * eta_c_arr + bw * w_c + bq * q_c
            eta_e_rec[0] = eta_e[0]
            for k in range(N_s - 1):
                eta_e_rec[k + 1] = phi_dt * eta_e_rec[k] + gam_dt * u_act[k]
        rec_label = "ηe reconstruit (ZOH)"
    rmse_rec = float(np.sqrt(np.mean((eta_e - eta_e_rec) ** 2)))

    # ── Layout : 8 panneaux data + 1 panneau texte matrices ──────────────────
    fig = plt.figure(figsize=(16, 23))
    gs  = fig.add_gridspec(9, 1, height_ratios=[1]*8 + [2], hspace=0.45)
    axes   = [fig.add_subplot(gs[i]) for i in range(8)]
    ax_mat = fig.add_subplot(gs[8])

    fig.suptitle(
        f"{title}\n"
        f"score={m.global_score:.3f}  ρ={result.model.spectral_radius:.4f}  "
        f"stable={result.model.is_stable}  |  "
        f"τ={tau:.4f}s   K={K:.4f}   bw={bw:.4f}   bq={bq:.4f}  |  "
        f"R²={r2:.3f}   RMSE={rmse:.4f}  |  "
        f"cbw={cbw:.1%}   cbq={cbq:.1%}",
        fontsize=8,
    )

    def _span(ax, i=99):
        ax.axvspan(trim.i0 * dt,     trim.i1 * dt,     color="limegreen", alpha=0.12,
                   label="Trim"        if i == 0 else None)
        ax.axvspan(trim.i0_avg * dt, trim.i1_avg * dt, color="orange",    alpha=0.18,
                   label="Avg window"  if i == 0 else None)
        ax.axvline(trim.i0 * dt, color="red", ls=":", lw=1.2, alpha=0.8,
                   label="Model start" if i == 0 else None)
        ax.grid(True, alpha=0.3)

    # Panneaux 0-3 : états aéro
    for i, (ax, lbl) in enumerate(zip(axes[:4], _STATE_LABELS[dyn])):
        ax.plot(t_full, X_full[:, i], "k-",  lw=0.8, alpha=0.55, label="Measured")
        if X_full_lp is not None:
            ax.plot(t_full, X_full_lp[:, i], color="green", lw=0.9, alpha=0.85,
                    label="Measured (passe-bas)" if i == 0 else None)
        ax.plot(t_sim,  Xsim[:, i],   "b--", lw=1.4,              label="Simulated")
        _span(ax, i)
        ax.set_ylabel(lbl, fontsize=9)
        ax.text(0.01, 0.96,
                f"R²={m.r2[i]:.3f}  Pearson={m.pearson[i]:.3f}  Fit%={m.fit_pct[i]:.3f}",
                transform=ax.transAxes, fontsize=8, color="navy", va="top")
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)

    # Panneau 4 : ηc mesurée (colonne pilote uniquement)
    axes[4].plot(t_sim, eta_c_arr, "r-", lw=1.2, label="ηc (colonne)")
    axes[4].set_ylabel("ηc centré", fontsize=9)
    axes[4].legend(fontsize=8, loc="upper right")
    _span(axes[4])

    # Panneau 5 : ηe mesurée vs ηe reconstruite (simulation couplée ou ZOH)
    axes[5].plot(t_sim, eta_e,     "k-",  lw=0.9, alpha=0.8, label="ηe mesurée")
    axes[5].plot(t_sim, eta_e_rec, "g--", lw=1.4,             label=rec_label)
    axes[5].set_ylabel("ηe centré (°)", fontsize=9)
    axes[5].legend(fontsize=8, loc="upper right")
    axes[5].text(0.01, 0.96, f"RMSE reconst. = {rmse_rec:.4f}",
                 transform=axes[5].transAxes, fontsize=8, color="darkgreen", va="top")
    _span(axes[5])

    # Panneau 6 : TAS
    axes[6].plot(t_sim, TAS_m, "k-",  lw=0.8, alpha=0.7, label="Measured")
    axes[6].plot(t_sim, TAS_s, "b--", lw=1.4,             label="Simulated")
    axes[6].set_ylabel("TAS (m/s)", fontsize=9)
    axes[6].legend(fontsize=8, loc="upper right")
    _span(axes[6])

    # Panneau 7 : α
    axes[7].plot(t_sim, alpha_m, "k-",  lw=0.8, alpha=0.7, label="Measured")
    axes[7].plot(t_sim, alpha_s, "b--", lw=1.4,             label="Simulated")
    axes[7].set_ylabel("α (deg)", fontsize=9)
    axes[7].set_xlabel("Time (s)", fontsize=9)
    _span(axes[7])

    # ── Panneau matrices A, B + paramètres actionneur ─────────────────────────
    ax_mat.axis("off")
    Ac = result.model.Ac
    Bc = result.model.Bc
    sn = ["u", "w", "q", "θ"]

    def _fmt_matrix(name, mat, row_names, col_names):
        hdr  = f"{name}        " + "  ".join(f"{c:>9}" for c in col_names)
        rows = [hdr]
        for i, rn in enumerate(row_names):
            vals = "  ".join(f"{mat[i, j]:+9.4f}" for j in range(mat.shape[1]))
            rows.append(f"  [{rn}]  {vals}")
        return "\n".join(rows)

    elev_col = data.config.col("elevator")
    nu = Bc.shape[1]
    in_names = [elev_col] + ([f"in{j}" for j in range(1, nu)] if nu > 1 else [])

    act_text = (
        f"Actionneur (NLS couplé):\n"
        f"  τ = {tau:.5f} s      K  = {K:.5f}\n"
        f"  bw = {bw:.5f}        bq = {bq:.5f}\n"
        f"  aτ = {a_tau:.5f}     bc = {b_c:.5f}\n"
        f"  R²  = {r2:.4f}       RMSE = {rmse:.5f}\n"
        f"  contrib bw = {cbw:.1%}    contrib bq = {cbq:.1%}"
    )
    full_text = (_fmt_matrix("Ac", Ac, sn, sn)
                 + "\n\n"
                 + _fmt_matrix("Bc", Bc, sn, in_names)
                 + "\n\n"
                 + act_text
                 + "\n\n"
                 + _excitation_str(m))
    ax_mat.text(0.01, 0.97, full_text, transform=ax_mat.transAxes,
                fontsize=7.5, family="monospace", va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
    return fig


def _fig_reconstruction(result: PipelineResult, title: str) -> plt.Figure:
    """Dispatch : avec coupling OLS si disponible, sinon figure standard."""
    coupling = _get_coupling(result)
    if coupling:
        return _fig_reconstruction_with_coupling(result, title, coupling)
    return _fig_reconstruction_standard(result, title)


def _filtered_command_overlay(data, dt: float, trim, n: int, raw_col: str, center_val: float) -> Optional[np.ndarray]:
    """
    Version filtrée de raw_col (même filtre de Hampel que celui appliqué dans
    generate_ideal_2311 : médiane + MAD locales, fenêtre 1 seconde centrée
    convertie en échantillons via dt — pas un nombre de points fixe —,
    gatée par use_accel_lowpass), recentrée sur center_val et découpée sur
    [trim.i0, trim.i0+n] comme les autres traces du panneau élévateur.
    None si le filtre est désactivé ou si raw_col est absente.
    """
    if not getattr(data, "use_accel_lowpass", False) or raw_col not in data.df.columns:
        return None
    raw_full  = data.df[raw_col].values.astype(float)
    window_n  = max(1, int(round(1.0 / dt)))
    filt_full = maneuver_quality_id.FlightData._hampel_filter(raw_full, window_n)
    return (filt_full - center_val)[trim.i0: trim.i0 + n]


def _fig_ideal_excitation(result: PipelineResult, title: str) -> plt.Figure:
    """
    5-panel figure: ideal excitation elevator vs real, then ideal vs real trajectory.
    Dispatches to generate_ideal_2311 or generate_ideal_phugoid based on result.maneuver_type.
    """
    mtype       = getattr(result, "maneuver_type", "2311")
    is_lateral  = result.data.config.dynamics == "lateral"
    assessor    = ManeuverQualityAssessor(result.model)

    coupling  = _get_coupling(result)
    col_col   = result.data.config.get_col("column") if coupling else None
    use_col   = coupling is not None and bool(col_col)

    if mtype == "phugoid":
        Uc_ideal, debug = assessor.generate_ideal_phugoid(result.data, result.trim)
        quality_val  = debug.get("quality_phugoid", float("nan"))
        quality_lbl  = "phugoid quality"
        ideal_lbl    = "Ideal phugoid"
        type_tag     = "Phugoid"
    elif mtype == "short_period":
        Uc_ideal, debug = assessor.generate_ideal_short_period(result.data, result.trim)
        quality_val  = debug.get("quality_short_period", float("nan"))
        quality_lbl  = "short period quality"
        ideal_lbl    = "Ideal short period"
        type_tag     = "Short Period"
    elif mtype == "dutch_roll":
        Uc_ideal, debug = assessor.generate_ideal_dutch_roll(result.data, result.trim)
        quality_val  = debug.get("quality_dutch_roll", float("nan"))
        quality_lbl  = "dutch roll quality"
        ideal_lbl    = "Ideal dutch roll (rudder)"
        type_tag     = "Dutch Roll"
    elif is_lateral:
        # WheelPedal (ailerons+rudder) : generate_ideal_2311 seul (col elevator par défaut) ne
        # veut rien dire ici -- generate_ideal_lateral_2311 construit les deux canaux ensemble
        # (ailerons en colonne 0, rudder en colonne 1 de Uc_ideal).
        Uc_ideal, debug = assessor.generate_ideal_lateral_2311(
            result.data, result.trim, soft_ramps=SOFT_RAMPS_2311, ramp_tau_s=RAMP_TAU_S,
        )
        quality_val  = debug.get("quality_2311", float("nan"))
        quality_lbl  = "2311 quality"
        ideal_lbl    = "Ideal 2311 (ailerons)"
        type_tag     = "2311"
    else:
        gen_kw = {"input_col": col_col} if use_col else {}
        gen_kw["soft_ramps"] = SOFT_RAMPS_2311
        gen_kw["ramp_tau_s"] = RAMP_TAU_S
        gen_kw["score_col"]  = "diff_elevator_1d"
        gen_kw["refine_onset_with_std"] = False
        Uc_ideal, debug = assessor.generate_ideal_2311(result.data, result.trim, **gen_kw)
        quality_val  = debug.get("quality_2311", float("nan"))
        quality_lbl  = "2311 quality"
        ideal_lbl    = "Ideal 2311 (colonne)" if use_col else "Ideal 2311"
        type_tag     = "2311"

    trim  = result.trim
    dt    = result.data.dt
    Xc    = result.Xc
    n     = min(len(Xc), len(result.Uc))
    t     = np.arange(trim.i0, trim.i0 + n) * dt
    Uc_ideal_seg = Uc_ideal[trim.i0:trim.i0 + n]   # (n, nu) — col signal si use_col

    # Colonne U contenant le signal idéalisé : rudder (index 1) en dynamique
    # latérale pour dutch_roll (cf. generate_ideal_dutch_roll), ailerons
    # (index 0) pour le 2311 latéral (cf. generate_ideal_lateral_2311),
    # élévateur (index 0) sinon.
    ideal_u_idx = 1 if mtype == "dutch_roll" else 0

    # ── Signal à afficher dans le panneau 0 ──────────────────────────────────
    if use_col and col_col in result.data.df.columns:
        col_raw    = result.data.df[col_col].values
        col_mean   = float(col_raw[trim.i0_avg:trim.i1_avg].mean())
        eta_c_real = (col_raw - col_mean)[trim.i0: trim.i0 + n]
        input_real = eta_c_real
        input_lbl  = "ηc réelle (colonne)"
        input_ylbl = "ηc centré"
        input_filt = _filtered_command_overlay(result.data, dt, trim, n, col_col, col_mean)
    elif mtype == "dutch_roll":
        input_real = result.Uc[:n, ideal_u_idx]
        input_lbl  = "Real rudder"
        input_ylbl = "Rudder"
        input_filt = None   # pas "l'élévateur" — rudder latéral, non concerné
    elif is_lateral:
        input_real = result.Uc[:n, 0]
        input_lbl  = "Real aileron (diff)"
        input_ylbl = "Aileron diff"
        input_filt = None   # pas d'équivalent filtré pour l'instant côté latéral
    else:
        input_real = result.Uc[:n, 0]
        input_lbl  = "Real elevator"
        input_ylbl = "Elevator (rad)"
        input_filt = _filtered_command_overlay(
            result.data, dt, trim, n, result.data.config.col("elevator"), float(trim.U_mean[0])
        )

    # ── Simulation avec entrée idéale ────────────────────────────────────────
    eta_e_id = None   # ηe idéal reconstruit via ZOH (None si non disponible)
    if use_col and coupling:
        # Propagation ZOH : ηc_ideal → ηe_ideal via modèle OLS actionneur
        a_tau = coupling.get("a_tau_act", float("nan"))
        b_c   = coupling.get("b_c_act",  float("nan"))
        bw    = coupling.get("bw",       float("nan"))
        bq    = coupling.get("bq",       float("nan"))
        if all(np.isfinite([a_tau, b_c, bw, bq])):
            phi_dt = float(np.exp(a_tau * dt))
            gam_dt = (phi_dt - 1.0) / a_tau if abs(a_tau) > 1e-12 else dt
            eta_c_id = Uc_ideal_seg[:n, 0]
            w_c = Xc[:n, 1];  q_c = Xc[:n, 2]
            u_act = b_c * eta_c_id + bw * w_c + bq * q_c
            eta_e_id = np.zeros(n)
            eta_e_id[0] = result.Uc[0, 0]
            for k in range(n - 1):
                eta_e_id[k + 1] = phi_dt * eta_e_id[k] + gam_dt * u_act[k]
            Uc_ideal_elev        = result.Uc[:n].copy()
            Uc_ideal_elev[:, 0]  = eta_e_id
        else:
            Uc_ideal_elev = result.Uc[:n].copy()
        Xsim_i = result.model.simulate_matching(Xc[:n], Uc_ideal_elev, dt)
    else:
        Xsim_i = result.model.simulate_matching(Xc[:n], Uc_ideal_seg[:n], dt)

    metrics_real, metrics_ideal = assessor.compare(
        result.Xc, result.Uc[:n], Uc_ideal_seg[:n] if not use_col else Uc_ideal_seg[:n],
        result.data.dt, dynamics=result.data.config.dynamics,
    )

    Xsim_r = result.model.simulate_matching(Xc[:n], result.Uc[:n], dt)

    status = debug.get("status", "?")

    show_eta_e  = use_col and eta_e_id is not None
    show_rudder = is_lateral and mtype != "dutch_roll"   # dutch_roll EST déjà le panneau rudder
    n_panels    = 6 if (show_eta_e or show_rudder) else 5
    fig, axes   = plt.subplots(n_panels, 1, figsize=(16, 2.8 * n_panels + 1), sharex=True)

    # Command score (test-card procedure compliance) : uniquement calculé par
    # generate_ideal_short_period, absent pour les autres types de manœuvre.
    command_line = ""
    if "command_score" in debug:
        cmd_score = debug.get("command_score", float("nan"))
        max_bank  = debug.get("max_bank_angle_deg", float("nan"))
        bank_ok   = debug.get("bank_angle_ok", True)
        bank_str  = f"{max_bank:.1f}°" if np.isfinite(max_bank) else "n/a"
        command_line = (
            f"\ncommand score={cmd_score:.3f}  (bank max={bank_str}, "
            f"{'OK' if bank_ok else 'VIOLATION >5°, essai à refaire'})"
        )

    fig.suptitle(
        f"{title} — Ideal {type_tag} vs Measured  [{status}]\n"
        f"Real score={metrics_real.global_score:.3f}  "
        f"Ideal score={metrics_ideal.global_score:.3f}  "
        f"{quality_lbl}={quality_val:.3f}  |  "
        f"trim [{trim.i0*dt:.1f}s – {trim.i1*dt:.1f}s]"
        f"{command_line}",
        fontsize=9,
    )

    def _add_trim_spans(ax, first=False):
        ax.axvspan(trim.i0 * dt, trim.i1 * dt,
                   color="limegreen", alpha=0.10, label="Trim" if first else None)
        ax.axvspan(trim.i0_avg * dt, trim.i1_avg * dt,
                   color="orange", alpha=0.15, label="Avg window" if first else None)
        ax.axvline(trim.i0 * dt, color="red", ls=":", lw=1.0, alpha=0.7,
                   label="Model start" if first else None)

    # Panneau 0 : ηc (ou élévateur/rudder) réelle vs idéale [vs filtrée]
    axes[0].plot(t, input_real,                      "k-",  lw=1.0, label=input_lbl)
    axes[0].plot(t, Uc_ideal_seg[:n, ideal_u_idx],    "r--", lw=1.4, label=ideal_lbl)
    if input_filt is not None:
        axes[0].plot(t, input_filt, "g-", lw=1.0, alpha=0.8, label=f"{input_lbl} (filtré)")

    # TrendAnalyzer segment spans: L=green, U=red, D=blue (phugoid only)
    _TREND_COLOR = {"L": "green", "U": "red", "D": "blue"}
    _TREND_LABEL = {"L": "Trend L", "U": "Trend U", "D": "Trend D"}
    _added_trend = set()
    for (s_abs, e_abs, trend) in debug.get("segments_summary", []):
        color = _TREND_COLOR.get(trend, "gray")
        label = _TREND_LABEL.get(trend, trend) if trend not in _added_trend else None
        if label:
            _added_trend.add(trend)
        axes[0].axvspan(s_abs * dt, (e_abs + 1) * dt,
                        color=color, alpha=0.18, label=label, zorder=0)

    _add_trim_spans(axes[0], first=True)
    axes[0].set_ylabel(input_ylbl, fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right", ncol=4)
    axes[0].grid(True, alpha=0.3)

    # Panneau 1 (optionnel) : ηe mesuré vs ηe idéal reconstruit par ZOH
    if show_eta_e:
        ax_e = axes[1]
        ax_e.plot(t, result.Uc[:n, 0], "k-",  lw=0.9, alpha=0.8, label="ηe mesurée")
        ax_e.plot(t, eta_e_id,          "g--", lw=1.4,             label="ηe idéale (ZOH ← ηc idéale)")
        _add_trim_spans(ax_e)
        ax_e.set_ylabel("ηe centré (°)", fontsize=9)
        ax_e.legend(fontsize=7, loc="upper right")
        ax_e.grid(True, alpha=0.3)
        state_axes = axes[2:]
    elif show_rudder:
        ax_r = axes[1]
        ax_r.plot(t, result.Uc[:n, 1],       "k-",  lw=1.0, label="Real rudder")
        ax_r.plot(t, Uc_ideal_seg[:n, 1],    "r--", lw=1.4, label="Ideal rudder")
        _add_trim_spans(ax_r)
        ax_r.set_ylabel("Rudder", fontsize=9)
        ax_r.legend(fontsize=7, loc="upper right")
        ax_r.grid(True, alpha=0.3)
        state_axes = axes[2:]
    else:
        state_axes = axes[1:]

    # Panneaux états
    for i, (ax, lbl) in enumerate(zip(state_axes, _STATE_LABELS[result.data.config.dynamics])):
        ax.plot(t, Xc[:n, i],       "k-",  lw=1.0, label="Measured")
        ax.plot(t, Xsim_r[:n, i],   "b--", lw=1.2, label="Sim (real ηc)")
        ax.plot(t, Xsim_i[:n, i],   "r-.", lw=1.2, label="Sim (ideal ηc)")
        _add_trim_spans(ax)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(True, alpha=0.3)

    state_axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    return fig


def _fig_ideal_excitation_elevator(result: PipelineResult, title: str) -> plt.Figure:
    """
    Page de comparaison : generate_ideal_2311 sur l'élévateur (mode classique),
    indépendamment du coupling OLS. Toujours 5 panneaux : ηe + 4 états.
    """
    mtype    = getattr(result, "maneuver_type", "2311")
    assessor = ManeuverQualityAssessor(result.model)

    if mtype == "phugoid":
        Uc_ideal, debug = assessor.generate_ideal_phugoid(result.data, result.trim)
        quality_val = debug.get("quality_phugoid", float("nan"))
        quality_lbl = "phugoid quality"
        type_tag    = "Phugoid"
    elif mtype == "short_period":
        Uc_ideal, debug = assessor.generate_ideal_short_period(result.data, result.trim)
        quality_val = debug.get("quality_short_period", float("nan"))
        quality_lbl = "short period quality"
        type_tag    = "Short Period"
    else:
        Uc_ideal, debug = assessor.generate_ideal_2311(
            result.data, result.trim,
            soft_ramps=SOFT_RAMPS_2311, ramp_tau_s=RAMP_TAU_S, score_col="diff_elevator_1d",
            refine_onset_with_std=False,
        )
        quality_val = debug.get("quality_2311", float("nan"))
        quality_lbl = "2311 quality"
        type_tag    = "2311"

    trim = result.trim
    dt   = result.data.dt
    Xc   = result.Xc
    n    = min(len(Xc), len(result.Uc))
    t    = np.arange(trim.i0, trim.i0 + n) * dt
    Uc_ideal_seg = Uc_ideal[trim.i0:trim.i0 + n]

    Xsim_r = result.model.simulate_matching(Xc[:n], result.Uc[:n],    dt)
    Xsim_i = result.model.simulate_matching(Xc[:n], Uc_ideal_seg[:n], dt)

    metrics_real, metrics_ideal = assessor.compare(
        result.Xc, result.Uc[:n], Uc_ideal_seg[:n],
        dt, dynamics=result.data.config.dynamics,
    )

    status = debug.get("status", "?")
    fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True)
    fig.suptitle(
        f"{title} — Ideal {type_tag} (élévateur) vs Measured  [{status}]\n"
        f"Real score={metrics_real.global_score:.3f}  "
        f"Ideal score={metrics_ideal.global_score:.3f}  "
        f"{quality_lbl}={quality_val:.3f}  |  "
        f"trim [{trim.i0*dt:.1f}s – {trim.i1*dt:.1f}s]",
        fontsize=9,
    )

    def _span(ax, first=False):
        ax.axvspan(trim.i0 * dt,     trim.i1 * dt,     color="limegreen", alpha=0.10,
                   label="Trim"        if first else None)
        ax.axvspan(trim.i0_avg * dt, trim.i1_avg * dt, color="orange",    alpha=0.15,
                   label="Avg window"  if first else None)
        ax.axvline(trim.i0 * dt, color="red", ls=":", lw=1.0, alpha=0.7,
                   label="Model start" if first else None)
        ax.grid(True, alpha=0.3)

    # Panneau 0 : ηe réelle vs ηe idéale vs ηe filtrée
    axes[0].plot(t, result.Uc[:n, 0],    "k-",  lw=1.0, label="ηe mesurée")
    axes[0].plot(t, Uc_ideal_seg[:n, 0], "r--", lw=1.4, label=f"Ideal {type_tag}")
    input_filt = _filtered_command_overlay(
        result.data, dt, trim, n, result.data.config.col("elevator"), float(trim.U_mean[0])
    )
    if input_filt is not None:
        axes[0].plot(t, input_filt, "g-", lw=1.0, alpha=0.8, label="ηe mesurée (filtré)")
    _TREND_COLOR = {"L": "green", "U": "red", "D": "blue"}
    _added_trend: set = set()
    for (s_abs, e_abs, trend) in debug.get("segments_summary", []):
        color = _TREND_COLOR.get(trend, "gray")
        label = trend if trend not in _added_trend else None
        if label:
            _added_trend.add(trend)
        axes[0].axvspan(s_abs * dt, (e_abs + 1) * dt, color=color, alpha=0.18,
                        label=label, zorder=0)
    _span(axes[0], first=True)
    axes[0].set_ylabel("ηe centré (°)", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right", ncol=4)

    # Panneaux 1-4 : états
    for i, (ax, lbl) in enumerate(zip(axes[1:], _STATE_LABELS[result.data.config.dynamics])):
        ax.plot(t, Xc[:n, i],       "k-",  lw=1.0, label="Measured")
        ax.plot(t, Xsim_r[:n, i],   "b--", lw=1.2, label="Sim (ηe réelle)")
        ax.plot(t, Xsim_i[:n, i],   "r-.", lw=1.2, label="Sim (ηe idéale)")
        _span(ax)
        ax.set_ylabel(lbl, fontsize=9)

    axes[1].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    return fig


def _fig_kde_scores(result: PipelineResult, title: str) -> plt.Figure:
    """
    Plot KDE scores from hyb for the full flight.
    One subplot per column of result.trim.scores_df.
    Threshold lines at 0.95 / 0.90, trim + averaging window highlighted.
    """
    scores_df = result.trim.scores_df
    trim = result.trim
    dt   = result.data.dt
    N    = len(result.data.df)
    t    = np.arange(N) * dt

    if scores_df is None or scores_df.empty:
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.text(0.5, 0.5, "Aucun score KDE disponible",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        fig.suptitle(f"{title} — KDE Scores", fontsize=9)
        return fig

    cols = list(scores_df.columns)[:10]   # max 10 panneaux
    n    = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"{title} — KDE Scores  [col utilisée : {result.trim.score_col}]\n"
        f"trim [{trim.i0*dt:.1f}s – {trim.i1*dt:.1f}s]  "
        f"avg [{trim.i0_avg*dt:.1f}s – {trim.i1_avg*dt:.1f}s]",
        fontsize=9,
    )

    for ax, col in zip(axes, cols):
        y = pd.to_numeric(scores_df[col], errors="coerce").to_numpy(float)
        # scores_df est normalement déjà à pleine résolution (TrimDetector le
        # ré-échantillonne dès qu'un downsample a servi à la détection de trim) ;
        # ce ré-échantillonnage reste un filet de sécurité, pour ne jamais faire
        # tomber la fenêtre de moyennage (indices pleine résolution) hors de y
        # via un simple padding NaN.
        if len(y) != N:
            y = np.interp(np.linspace(0, len(y) - 1, N), np.arange(len(y)), y)

        is_combined = col == "_COMBINED_" or str(col).lower().startswith("combined")
        color = "navy" if is_combined else "steelblue"
        lw    = 1.8  if is_combined else 1.0

        ax.plot(t, y, color=color, lw=lw)
        ax.axhline(0.95, color="red",    ls="--", lw=0.8, alpha=0.7)
        ax.axhline(0.90, color="orange", ls="--", lw=0.8, alpha=0.5)
        ax.axvspan(trim.i0     * dt, trim.i1     * dt, color="limegreen", alpha=0.12)
        ax.axvspan(trim.i0_avg * dt, trim.i1_avg * dt, color="orange",    alpha=0.20)

        mean_trim = float(np.nanmean(y[trim.i0_avg:trim.i1_avg]))
        ax.set_ylabel(col, fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.text(0.01, 0.95, f"moy_trim = {mean_trim:.3f}",
                transform=ax.transAxes, fontsize=7, color="navy", va="top")

    # Légende sur le premier panneau seulement
    axes[0].axhline(0.95, color="red",       ls="--", lw=0.8, label="seuil 0.95")
    axes[0].axhline(0.90, color="orange",    ls="--", lw=0.8, label="seuil 0.90")
    axes[0].axvspan(0, 0, color="limegreen", alpha=0.3,  label="Trim")
    axes[0].axvspan(0, 0, color="orange",    alpha=0.35, label="Avg window")
    axes[0].legend(fontsize=7, loc="upper right", ncol=4)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# AIAA KDE score subfigures (3 in × 1 in, one file per score column)
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib as _mpl
_mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         12,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth":   0.9,
    "figure.dpi":        300,
})

_AIAA_FIG_W = 3.0   # in
_AIAA_FIG_H = 1.0   # in


def save_aiaa_kde_subfigures(result: PipelineResult, prefix: str, out_dir: str) -> None:
    """Save one AIAA subfigure (3×1 in) per KDE score column."""
    scores_df = result.trim.scores_df
    if scores_df is None or scores_df.empty:
        return

    dt   = result.data.dt
    N    = len(result.data.df)
    t    = np.arange(N) * dt

    os.makedirs(out_dir, exist_ok=True)

    for col in scores_df.columns:
        y = pd.to_numeric(scores_df[col], errors="coerce").to_numpy(float)
        if len(y) < N:
            y = np.pad(y, (0, N - len(y)), constant_values=np.nan)
        elif len(y) > N:
            y = y[:N]

        is_combined = col == "_COMBINED_" or str(col).lower().startswith("combined")
        color = "navy" if is_combined else "steelblue"

        fig, ax = plt.subplots(figsize=(_AIAA_FIG_W, _AIAA_FIG_H))
        ax.plot(t, y, color=color, linewidth=0.9)
        ax.axhline(0.95, color="red",    ls="--", lw=0.7, alpha=0.8)
        ax.axhline(0.90, color="orange", ls="--", lw=0.7, alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=12)
        ax.grid(True, linewidth=0.35, color="0.80", linestyle="--")
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        fig.tight_layout(pad=0.2)

        safe_col = col.replace("_", "").replace(" ", "_")[:20]
        fname = f"{prefix}_{safe_col}"
        for ext in ("pdf", "png"):
            fpath = os.path.join(out_dir, f"{fname}.{ext}")
            try:
                fig.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0.08)
            except PermissionError:
                print(f"  WARNING: {fpath} locked — skipped")
        plt.close(fig)
    print(f"  AIAA KDE subfigures saved → {out_dir}")


_AIAA_SIGNAL_COLS = [
    'l04_sysi_elv_avg_f8',
    'g04_eom_q_deg_f8',
    'g04_eom_alpha_f8_uncal',
    'l04_sysi_ail_f8(1)',
    'l04_sysi_ail_f8(2)',
]
_AIAA_SIGNAL_SHORT = {
    'l04_sysi_elv_avg_f8':   'elevator',
    'g04_eom_q_deg_f8':      'pitch_rate',
    'g04_eom_alpha_f8_uncal':'alpha',
    'l04_sysi_ail_f8(1)':    'aileron_L',
    'l04_sysi_ail_f8(2)':    'aileron_R',
}
_SCORE_COLORS = [
    (0.95, '#4CAF50'),   # vert   ≥ 0.92
    (0.94, '#FFEB3B'),   # jaune  ≥ 0.90
    (0.9250, '#FF9800'),   # orange ≥ 0.85
    (0.00, '#F44336'),   # rouge  < 0.85
]

def _score_color(s: float) -> str:
    if np.isnan(s):
        return _SCORE_COLORS[-1][1]
    for thresh, col in _SCORE_COLORS:
        if s >= thresh:
            return col
    return _SCORE_COLORS[-1][1]


def save_aiaa_colored_signals(result: PipelineResult, prefix: str, out_dir: str) -> None:
    """Save one AIAA subfigure per signal, colored by combined KDE score."""
    from matplotlib.collections import LineCollection

    scores_df = result.trim.scores_df
    if scores_df is None or scores_df.empty:
        return

    df = result.data.df
    dt = result.data.dt
    N  = len(df)
    t  = np.arange(N) * dt

    # Pick the best combined score column
    for cand in ("_COMBINED_", "combined_wmean", "combined", "score"):
        if cand in scores_df.columns:
            score_col = cand
            break
    else:
        score_col = scores_df.columns[0]

    # The scorer may run at a different sample rate than result.data.df.
    # Interpolate the M scores onto the N-sample signal time grid.
    vals = pd.to_numeric(scores_df[score_col], errors="coerce").to_numpy(float)
    M    = len(vals)
    # Map score sample i → signal sample round(i * N/M)
    score = np.interp(np.arange(N), np.linspace(0, N - 1, M), vals)

    os.makedirs(out_dir, exist_ok=True)

    for col in _AIAA_SIGNAL_COLS:
        if col not in df.columns:
            continue
        y = df[col].to_numpy(float)

        # Build per-segment colors from score
        points   = np.array([t, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors   = [_score_color(score[i]) for i in range(len(segments))]

        fig, ax = plt.subplots(figsize=(_AIAA_FIG_W, _AIAA_FIG_H))
        lc = LineCollection(segments, colors=colors, linewidth=0.9)
        ax.add_collection(lc)

        ymin, ymax = np.nanmin(y), np.nanmax(y)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(ymin - 1, ymax + 1)

        ax.tick_params(labelsize=12)
        ax.grid(True, linewidth=0.35, color='0.80', linestyle='--')
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        fig.tight_layout(pad=0.2)

        short  = _AIAA_SIGNAL_SHORT.get(col, col)
        fname  = f"{prefix}_{short}"
        for ext in ('pdf', 'png'):
            fpath = os.path.join(out_dir, f"{fname}.{ext}")
            try:
                fig.savefig(fpath, dpi=300, bbox_inches='tight', pad_inches=0.08)
            except PermissionError:
                print(f"  WARNING: {fpath} locked — skipped")
        plt.close(fig)
    print(f"  AIAA colored signals saved → {out_dir}")


_COMMAND_KEYS = ["elevator", "aileron_l", "aileron_r", "rudder"]


def _fig_commands(result: PipelineResult, title: str) -> plt.Figure:
    """Plot all raw control inputs (elevator, ailerons, rudder) + body accelerations (ax, az,
    avec passe-bas superposé — colonnes déjà calculées par FlightData.preprocess)."""
    data = result.data
    trim = result.trim
    dt   = data.dt
    df   = data.df

    panels = []
    for key in _COMMAND_KEYS:
        try:
            col = data.config.col(key)
        except (KeyError, Exception):
            continue
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        panels.append((key, col, y, None))

    # N1% (moyenne moteurs, normalisée [0,1] dans _n1p, cf. preprocess étape
    # 5) — n'est incluse comme 2e entrée du modèle (nu=2, cf. FlightData.U)
    # que si data._n1_active (variation sur TOUT le fichier > seuil) ; ce
    # statut est affiché dans le titre du panneau pour voir directement si
    # un mouvement N1% pendant la manœuvre a été raté par ce critère global.
    if "_n1p" in df.columns:
        y_n1p     = pd.to_numeric(df["_n1p"], errors="coerce").to_numpy(float) * 100.0
        n1_active = getattr(data, "_n1_active", False)
        panels.append(("N1%", f"_n1p — actif dans le modèle (nu=2) : {'oui' if n1_active else 'non'}", y_n1p, None))

    # Accélérations corps (m/s², cf. FlightData.preprocess étape 6) plutôt que N1/colonne,
    # avec passe-bas superposé — pris directement depuis "{col}_lp" (même
    # filtre que celui utilisé par la reconstruction quand
    # USE_ACCEL_LOWPASS_RECON est activé), pas recalculé ici.
    fc_lp = getattr(data, "accel_lowpass_fc_hz", ACCEL_LOWPASS_FC_HZ)
    for key, col in (("ax", "_ax"), ("az", "_az")):
        if col in df.columns:
            y_acc = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            lp_col = f"{col}_lp"
            y_lp = pd.to_numeric(df[lp_col], errors="coerce").to_numpy(float) if lp_col in df.columns else None
            panels.append((key, col, y_acc, y_lp))

    if not panels:
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.text(0.5, 0.5, "Aucune colonne de commande trouvée",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        fig.suptitle(f"{title} — Commandes", fontsize=9)
        return fig

    n = len(panels)
    t = np.arange(len(df)) * dt

    # Spectre de Fourier (ax, az) sur le segment post-trim, en plus des
    # tracés temporels — panneaux séparés (axe fréquence, pas de sharex avec
    # le reste) ajoutés en bas de page.
    fft_cols = [(key, col) for key, col in (("ax", "_ax"), ("az", "_az")) if col in df.columns]
    n_fft = len(fft_cols)

    fig = plt.figure(figsize=(16, 2.2 * n + 2.4 * n_fft + 1))
    gs  = fig.add_gridspec(n + n_fft, 1, hspace=0.5)
    axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, n):
        axes.append(fig.add_subplot(gs[i, 0], sharex=axes[0]))

    fig.suptitle(
        f"{title} — Commandes réelles\n"
        f"trim [{trim.i0*dt:.1f}s – {trim.i1*dt:.1f}s]  "
        f"avg [{trim.i0_avg*dt:.1f}s – {trim.i1_avg*dt:.1f}s]",
        fontsize=9,
    )

    for i, (ax, (key, col, y, y_lp)) in enumerate(zip(axes, panels)):
        ax.plot(t, y, "k-", lw=0.9, label="mesuré" if y_lp is not None else None)
        if y_lp is not None:
            ax.plot(t, y_lp, "-", color="green", lw=1.3,
                    label=f"passe-bas {fc_lp:g} Hz")
            ax.legend(fontsize=7, loc="upper right")
        ax.axvspan(trim.i0     * dt, trim.i1     * dt, color="limegreen", alpha=0.12,
                   label="Trim" if i == 0 else None)
        ax.axvspan(trim.i0_avg * dt, trim.i1_avg * dt, color="orange",    alpha=0.18,
                   label="Avg window" if i == 0 else None)
        ax.axvline(trim.i0     * dt, color="red", ls=":", lw=1.0, alpha=0.7,
                   label="Model start" if i == 0 else None)
        ax.set_ylabel(key, fontsize=9)
        ax.set_title(col, fontsize=7, pad=2)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper right", ncol=3)
    axes[-1].set_xlabel("Time (s)", fontsize=9)

    # ── Spectres FFT(ax), FFT(az) sur le segment post-trim ─────────────────
    i_start = int(trim.i1_avg)
    for j, (key, col) in enumerate(fft_cols):
        ax_fft = fig.add_subplot(gs[n + j, 0])
        y_seg = pd.to_numeric(df[col], errors="coerce").to_numpy(float)[i_start:]
        y_seg = y_seg[np.isfinite(y_seg)]
        color = "navy" if key == "ax" else "darkred"
        if len(y_seg) < 10:
            ax_fft.text(0.5, 0.5, "segment trop court pour la FFT",
                        ha="center", va="center", transform=ax_fft.transAxes, fontsize=9)
        else:
            y_d = y_seg - np.mean(y_seg)
            N_s = len(y_d)
            freqs = np.fft.rfftfreq(N_s, dt)
            mag   = np.abs(np.fft.rfft(y_d)) * 2.0 / N_s
            ax_fft.semilogy(freqs[1:], mag[1:], color=color, lw=1.0)
            ax_fft.set_xlim(0.0, min(2.0, freqs[-1] if len(freqs) > 1 else 2.0))
            pk = 1 + int(np.argmax(mag[1:])) if N_s > 4 else 0
            if pk > 0:
                ax_fft.axvline(freqs[pk], color="green", ls=":", lw=1.0, alpha=0.8)
                ax_fft.set_title(
                    f"pic dominant : {freqs[pk]:.4f} Hz (T={1.0/freqs[pk]:.1f} s)",
                    fontsize=7.5, pad=2,
                )
        ax_fft.set_ylabel(f"|FFT({key})| (m/s²)", fontsize=9)
        ax_fft.grid(True, which="both", alpha=0.3)
        if j == n_fft - 1:
            ax_fft.set_xlabel("Fréquence (Hz)", fontsize=9)

    # tight_layout() ne gère pas bien ce gridspec mixte (panneaux temporels
    # sharex + panneaux fréquentiels indépendants) : marge fixe à la place.
    fig.subplots_adjust(top=0.95, bottom=0.03, hspace=0.55)
    return fig



# ─────────────────────────────────────────────────────────────────────────────
# FRF MIMO (pyFRF) — fréquences propres et déformées modales vs modèle grey-box
# ─────────────────────────────────────────────────────────────────────────────
def _compute_mimo_frf_raw(Uc: np.ndarray, Xc: np.ndarray, dt: float) -> Optional[tuple]:
    """
    Estime la FRF MIMO (entrées Uc -> états Xc) par l'estimateur H1 de pyFRF
    (cf. https://pyfrf.readthedocs.io/en/latest/Showcase.html#MIMO-systems-and-averaging),
    moyennée sur des segments de Welch recouvrants à 50 % — un seul essai est
    disponible par fichier, donc l'"averaging" vient du découpage en segments
    et non de mesures répétées.

    resp_type="d" donne un facteur de conversion identité côté pyFRF (pas de
    mise à l'échelle en jω) : H1 est le ratio brut Xc/Uc, directement
    comparable à la FRF analytique du modèle d'état (Ac, Bc).

    Retourne (freq_hz, H1, coherence) avec H1 de forme (nx, nu, n_freq) et
    coherence de forme (nx, n_freq) — cette dernière sert à écarter les
    fréquences où l'entrée n'a quasiment pas excité le système (H1 y devient
    un ratio bruit/bruit sans signification physique), ou None si le segment
    est trop court pour être découpé en segments exploitables.

    Prend Uc/Xc/dt bruts (pas un PipelineResult) : réutilisable indépendamment
    de la page de diagnostic _fig_frf_mimo, qui est son seul appelant actuel.
    """
    n = min(len(Uc), len(Xc))
    if n < 128:
        return None
    exc  = np.ascontiguousarray(Uc[:n]).T[np.newaxis, :, :]   # (1, nu, n)
    resp = np.ascontiguousarray(Xc[:n]).T[np.newaxis, :, :]   # (1, nx, n)
    nperseg = int(np.clip(n // 6, 64, n))
    frf = FRF(
        sampling_freq=int(round(1.0 / dt)),
        exc=exc, resp=resp,
        exc_type="f", resp_type="d",
        window="hann", nperseg=nperseg, noverlap=nperseg // 2,
        frf_type="H1",
    )
    coherence = np.clip(np.real(frf.get_coherence()), 0.0, 1.0)   # (nx, n_freq)
    return frf.get_f_axis(), frf.get_H1(), coherence


def _compute_mimo_frf(result: PipelineResult) -> Optional[tuple]:
    """FRF MIMO du résultat déjà ajusté — cf. _compute_mimo_frf_raw."""
    return _compute_mimo_frf_raw(Uc=result.Uc, Xc=result.Xc, dt=result.data.dt)


def _model_analytical_frf(Ac: np.ndarray, Bc: np.ndarray, freq_hz: np.ndarray) -> np.ndarray:
    """FRF analytique continue du modèle grey-box : H(f) = (j2πf I - Ac)^{-1} Bc."""
    nx = Ac.shape[0]
    I = np.eye(nx)
    H = np.empty((nx, Bc.shape[1], len(freq_hz)), dtype=complex)
    for k, f in enumerate(freq_hz):
        H[:, :, k] = np.linalg.solve(1j * 2.0 * np.pi * f * I - Ac, Bc)
    return H


def _model_modes(Ac: np.ndarray) -> list[dict]:
    """
    Fréquence propre, amortissement et déformée modale par paire de valeurs
    propres complexes conjuguées de Ac (les modes non oscillants, valeur
    propre réelle, sont ignorés — un pic-pointage sur la FRF ne peut de toute
    façon pas les révéler).
    """
    vals, vecs = np.linalg.eig(Ac)
    modes: list[dict] = []
    seen: set[int] = set()
    for i, lam in enumerate(vals):
        if i in seen or abs(lam.imag) < 1e-6:
            continue
        seen.add(i)
        for j in range(i + 1, len(vals)):
            if j not in seen and abs(vals[j] - np.conj(lam)) < 1e-6 * max(1.0, abs(lam)):
                seen.add(j)
                break
        wn = abs(lam)
        modes.append({
            "f_hz": wn / (2.0 * np.pi),
            "zeta": -lam.real / wn if wn > 1e-12 else float("nan"),
            "shape": vecs[:, i],
        })
    modes.sort(key=lambda m: m["f_hz"])
    return modes


def _frf_mode_near(freq_hz: np.ndarray, H1: np.ndarray, coherence: np.ndarray,
                    f_center: float, exc_idx: int = 0, octave: float = 0.8,
                    f_floor: float = 0.02, coh_min: float = 0.5) -> Optional[dict]:
    """
    Cherche, dans une bande d'environ ±1 octave autour de la fréquence propre
    f_center prédite par le modèle, le maximum de la fonction indicatrice de
    mode (somme des |H1|² sur les sorties, entrée élévateur) parmi les
    fréquences où la cohérence moyenne dépasse coh_min.

    La recherche est ancrée sur la prédiction du modèle plutôt qu'un
    pic-pointage global aveugle : un mode lourdement amorti (cas fréquent du
    court-période identifié ici) produit une réponse large sans maximum
    local net, qu'un pic-pointage aveugle sur toute la bande peut manquer
    complètement même quand la FRF empirique corrobore bien la zone. Rend
    None si aucune fréquence de la bande n'atteint coh_min (mode jugé non
    résolu par cette manœuvre).
    """
    f_lo = max(f_floor, f_center / (2.0 ** octave))
    f_hi = f_center * (2.0 ** octave)
    coh_mean = np.mean(coherence, axis=0)
    mask = (freq_hz >= f_lo) & (freq_hz <= f_hi) & (coh_mean >= coh_min)
    if not np.any(mask):
        return None
    idx_valid = np.flatnonzero(mask)
    mif = np.sum(np.abs(H1[:, exc_idx, :]) ** 2, axis=0)
    k = int(idx_valid[np.argmax(mif[idx_valid])])
    return {"f_hz": float(freq_hz[k]), "shape": H1[:, exc_idx, k],
            "coherence": float(coh_mean[k])}


def _fig_frf_mimo(result: PipelineResult, title: str) -> plt.Figure:
    """
    FRF MIMO (pyFRF, estimateur H1) : entrées Uc (élévateur [+N1]) -> états Xc.
    Pour chaque mode oscillant du modèle grey-box identifié (valeurs/vecteurs
    propres de Ac), cherche dans la FRF empirique — sur une bande d'environ
    ±1 octave autour de sa fréquence propre — la fréquence et la déformée
    modale correspondantes, avec la FRF analytique du modèle en superposition
    sur chaque canal.
    """
    dyn    = result.data.config.dynamics
    labels = _STATE_LABELS.get(dyn, _STATE_LABELS["longitudinal"])
    Ac, Bc = result.model.Ac, result.model.Bc
    nx     = Ac.shape[0]

    frf_data = _compute_mimo_frf(result)
    if frf_data is None:
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.text(0.5, 0.5, "Segment trop court pour une analyse FRF",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        fig.suptitle(f"{title} — FRF MIMO (pyFRF)", fontsize=9)
        return fig

    freq_hz, H1, coherence = frf_data
    H_model = _model_analytical_frf(Ac, Bc, freq_hz)
    coh_mean = np.mean(coherence, axis=0)

    duration_s  = min(len(result.Uc), len(result.Xc)) * result.data.dt
    model_modes = _model_modes(Ac)
    COH_MIN = 0.5
    pairs: list[tuple[dict, Optional[dict]]] = [
        (m, _frf_mode_near(freq_hz, H1, coherence, m["f_hz"],
                            f_floor=freq_hz[1] * 0.5, coh_min=COH_MIN))
        for m in model_modes
    ]

    n_modes_plot = min(2, len(pairs))
    fig = plt.figure(figsize=(16, 3.0 * nx + 6))
    gs  = fig.add_gridspec(nx + 2, max(n_modes_plot, 1),
                           height_ratios=[1] * nx + [1.4, 1.2], hspace=0.55, wspace=0.3)

    # La FRF n'a de contenu physique que près des fréquences propres du modèle ;
    # au-delà, |H_modèle| décroît de plusieurs décades (système strictement
    # propre) et noierait l'échelle y en log si on affichait tout le Nyquist.
    f_plot_max = float(np.clip(max((m["f_hz"] for m in model_modes), default=1.0) * 8.0,
                               3.0, freq_hz[-1]))
    pm = freq_hz <= f_plot_max
    freq_p, H1_p, H_model_p, coh_mean_p = freq_hz[pm], H1[:, :, pm], H_model[:, :, pm], coh_mean[pm]

    # zones de faible cohérence (H1 peu fiable), regroupées en segments contigus
    # puis fusionnées si séparées de moins de quelques bins (sinon la cohérence
    # bin-à-bin bruitée donne un aspect "code-barres" illisible).
    low_coh = coh_mean_p < COH_MIN
    coh_spans, start_k = [], None
    for k, flag in enumerate(low_coh):
        if flag and start_k is None:
            start_k = k
        elif not flag and start_k is not None:
            coh_spans.append((freq_p[start_k], freq_p[k - 1]))
            start_k = None
    if start_k is not None:
        coh_spans.append((freq_p[start_k], freq_p[-1]))
    gap_tol = 3.0 * (freq_hz[1] - freq_hz[0])
    merged_spans: list[tuple[float, float]] = []
    for s0, s1 in coh_spans:
        if merged_spans and s0 - merged_spans[-1][1] <= gap_tol:
            merged_spans[-1] = (merged_spans[-1][0], s1)
        else:
            merged_spans.append((s0, s1))

    # ── Panneaux 0..nx-1 : |FRF| empirique vs modèle, par état ────────────────
    for i, lbl in enumerate(labels):
        ax = fig.add_subplot(gs[i, :])
        for j, (s0, s1) in enumerate(merged_spans):
            ax.axvspan(s0, s1, color="0.85", alpha=0.5, zorder=0,
                       label=f"cohérence < {COH_MIN:g}" if (i == 0 and j == 0) else None)
        mag_emp = np.abs(H1_p[i, 0, :]) + 1e-12
        mag_mod = np.abs(H_model_p[i, 0, :]) + 1e-12
        ax.semilogy(freq_p, mag_emp, color="steelblue", lw=1.1,
                    label="FRF empirique (pyFRF, H1)")
        ax.semilogy(freq_p, mag_mod, color="navy", lw=1.3, ls="--",
                    label="FRF modèle (grey-box, analytique)")
        for m, p in pairs:
            ax.axvline(m["f_hz"], color="darkred", ls=":", lw=1.0, alpha=0.7)
            if p is not None:
                ax.axvline(p["f_hz"], color="darkorange", ls=":", lw=1.0, alpha=0.7)
        # une éventuelle antirésonance (|H| -> 0) ne doit pas comprimer tout
        # le reste de la courbe : on fixe la plage y à ~5 décades sous le pic
        # plutôt que de suivre le minimum réel (souvent un artefact de notch).
        y_top = max(mag_emp.max(), mag_mod.max())
        ax.set_ylim(y_top * 1e-5, y_top * 3)
        ax.set_ylabel(f"|H| {lbl}", fontsize=8.3)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(freq_p[0], freq_p[-1])
        if i == 0:
            ax.legend(fontsize=7.3, loc="upper right")
        if i == nx - 1:
            ax.set_xlabel("Fréquence (Hz)", fontsize=9)

    fig.suptitle(
        f"{title} — FRF MIMO (pyFRF, H1) vs modèle grey-box  |  entrée : élévateur → η_e (+N1 si actif)\n"
        f"durée exploitée = {duration_s:.1f}s  →  résolution ≈ {1.0 / duration_s:.4f} Hz  |  "
        f"recherche par mode : ±1 octave autour de f_modèle, cohérence ≥ {COH_MIN:g}",
        fontsize=9,
    )

    # ── Panneau texte : comparaison fréquences propres / amortissement ────────
    ax_txt = fig.add_subplot(gs[nx, :])
    ax_txt.axis("off")
    lines = [f"{'Mode':<6}{'f_modèle (Hz)':<15}{'f_FRF (Hz)':<13}{'Δf (%)':<10}{'cohérence':<11}{'ζ_modèle':<10}"]
    if not pairs:
        lines.append("  (aucun mode oscillant complexe trouvé dans le modèle identifié)")
    for k, (m, p) in enumerate(pairs, start=1):
        if p is not None and m["f_hz"] > 1e-9:
            df_pct = 100.0 * (p["f_hz"] - m["f_hz"]) / m["f_hz"]
            lines.append(f"{k:<6}{m['f_hz']:<15.4f}{p['f_hz']:<13.4f}{df_pct:<10.1f}{p['coherence']:<11.3f}{m['zeta']:<10.4f}")
        else:
            lines.append(f"{k:<6}{m['f_hz']:<15.4f}{'non résolu':<13}{'--':<10}{'--':<11}{m['zeta']:<10.4f}")
    ax_txt.text(0.01, 0.92, "\n".join(lines), transform=ax_txt.transAxes, fontsize=8.3,
                family="monospace", va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    # ── Panneaux déformées modales : modèle vs FRF, par mode apparié ──────────
    x = np.arange(nx)
    width = 0.35
    for k in range(n_modes_plot):
        m, p = pairs[k]
        ax_m = fig.add_subplot(gs[nx + 1, k])
        model_ref = m["shape"][np.argmax(np.abs(m["shape"]))]
        ax_m.bar(x - width / 2, np.real(m["shape"] / model_ref), width,
                 label="modèle (grey-box)", color="navy")
        if p is not None:
            frf_ref = p["shape"][np.argmax(np.abs(p["shape"]))]
            ax_m.bar(x + width / 2, np.real(p["shape"] / frf_ref), width,
                     label="FRF (pyFRF)", color="darkorange")
        ax_m.set_xticks(x)
        ax_m.set_xticklabels([lbl.split(" ")[0] for lbl in labels])
        ax_m.axhline(0, color="0.5", lw=0.6)
        ax_m.set_title(f"Mode {k + 1} : f ≈ {m['f_hz']:.3f} Hz", fontsize=8.5)
        ax_m.grid(True, alpha=0.3)
        if k == 0:
            ax_m.legend(fontsize=7.3)

    return fig


def _fig_alpha_elevator_sp_tf(result: PipelineResult, title: str) -> plt.Figure:
    """
    Bode (magnitude + phase) de la TF analytique classique du court-période
    (2 états α, q), tracée directement en fonction de s = jω :

        α(s)/δe(s) = (1/U1) [Zδe s + (Mδe U1 - Mq Zδe)]
                     / [s² - (Mq + Zα/U1 + Mα) s + (Zα Mq/U1 - Mα)]

    Les dérivées sont extraites du modèle grey-box continu ajusté (Ac, Bc,
    état [u, w, q, θ]) :
        Zα/U1 = Ac[1,1]      Mq  = Ac[2,2]      Mα  = Ac[2,1] · U1
        Zδe   = Bc[1,0]      Mδe = Bc[2,0]      U1  = trim.X_mean[0]
    (Zα/U1 et Zδe s'obtiennent directement puisque ẇ = U1·α̇ et w = U1·α
    font disparaître le facteur U1 dans la ligne ẇ du modèle d'état.)
    """
    from scipy.signal import TransferFunction, BadCoefficients

    dyn = result.data.config.dynamics
    if dyn != "longitudinal":
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.text(0.5, 0.5, "TF α/δe court-période : dynamique longitudinale requise",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        fig.suptitle(f"{title} — α/δe (TF court-période)", fontsize=9)
        return fig

    trim = result.trim
    Ac, Bc = result.model.Ac, result.model.Bc
    U1 = float(trim.X_mean[0])

    Za_over_U1 = float(Ac[1, 1])
    Mq         = float(Ac[2, 2])
    Ma         = float(Ac[2, 1]) * U1
    Zde        = float(Bc[1, 0])
    Mde        = float(Bc[2, 0])

    num = [Zde / U1, Mde - Mq * Zde / U1]
    den = [1.0, -(Mq + Za_over_U1 + Ma), Za_over_U1 * Mq - Ma]

    # Le numérateur est parfois quasi-nul (Zδe petit) → scipy avertit sur le
    # conditionnement (BadCoefficients) alors que le tracé reste correct
    # (juste un gain très faible aux basses fréquences) ; sans effet ici.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=BadCoefficients)
        tf = TransferFunction(num, den)
        w, mag_db, phase_deg = tf.bode(n=500)
    freq_hz = w / (2.0 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True)

    axes[0].semilogx(freq_hz, mag_db, color="navy", lw=1.4)
    axes[0].set_ylabel("|α(jω)/δe(jω)| (dB)", fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(freq_hz, phase_deg, color="darkred", lw=1.4)
    axes[1].set_ylabel("Phase (deg)", fontsize=9)
    axes[1].set_xlabel("Fréquence (Hz)", fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"{title} — α/δe (TF court-période analytique, Bode)\n"
        f"U1={U1:.2f} m/s | Zα/U1={Za_over_U1:.4f} 1/s | Mα={Ma:.4f} 1/s² | "
        f"Mq={Mq:.4f} 1/s | Zδe={Zde:.4f} (m/s²)/rad | Mδe={Mde:.4f} 1/s²/rad",
        fontsize=9,
    )

    return fig



def export_pdf(result: PipelineResult, pdf_path: str) -> None:
    """Save commands + trim scores + reconstruction + excitation/actuator figures to a single PDF."""
    fname = os.path.basename(result.file_path)
    if _get_coupling(result):
        fig_fns = (_fig_ideal_excitation, _fig_commands, _fig_kde_scores, _fig_reconstruction,
                   _fig_ideal_excitation_elevator)
    else:
        fig_fns = (_fig_ideal_excitation, _fig_commands, _fig_kde_scores, _fig_reconstruction)
    if EXPORT_FRF:
        fig_fns = fig_fns + (_fig_alpha_elevator_sp_tf,)
    with PdfPages(pdf_path) as pdf:
        for fig_fn in fig_fns:
            fig = None
            try:
                fig = fig_fn(result, title=fname)
                pdf.savefig(fig)
            except Exception as exc:
                print(f"    [WARN] plot failed ({fig_fn.__name__}): {exc}")
                traceback.print_exc()
            finally:
                if fig is not None:
                    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Row builder
# ─────────────────────────────────────────────────────────────────────────────
def _result_to_row(result: PipelineResult, folder_name: str) -> dict:
    """Flatten PipelineResult into a flat dict for the Excel output."""
    m        = result.metrics
    trim     = result.trim
    mdl      = result.model
    dynamics = result.data.config.dynamics

    # ── Trim flight condition ─────────────────────────────────────────────────
    if dynamics == "lateral":
        sl     = slice(trim.i0_avg, trim.i1_avg)
        df     = result.data.df
        u0_ms  = float(np.nanmean(df["_u"].values[sl]))
        theta0 = float(np.nanmean(df[result.data.config.col("theta")].values[sl]))
        trim_states = {
            "v0_ms":      float(trim.X_mean[0]),
            "u0_ms":      u0_ms,
            "theta0_deg": float(np.rad2deg(theta0)),
            "phi0_deg":   float(np.rad2deg(trim.X_mean[3])),
        }
    else:
        trim_states = {
            "u0_ms":      float(trim.X_mean[0]),
            "theta0_deg": float(np.rad2deg(trim.X_mean[3])),
        }

    row: dict = {
        "Folder":        folder_name,
        "File":          os.path.basename(result.file_path),
        "Aircraft":      f"Aircraft{result.data.config.plane}",
        "dynamics":      dynamics,
        "maneuver_type": getattr(result, "maneuver_type", "2311"),
        "mach":     trim.mach_mean,
        "alpha_deg": np.rad2deg(trim.alpha_mean),
        **trim_states,
        "trim_i0":  trim.i0,
        "trim_i1":  trim.i1,
        "n_fit":    getattr(mdl, "_n_fit", None),
        "nu":       mdl.nu,
        "stable":   mdl.is_stable,
        "rho":      mdl.spectral_radius,
        **m.summary(),   # global_score + r2/pearson/fit_pct/rmse + derivatives + trim_kde + 2311
    }
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def _run_for_aircraft(plane: int, root_dir: str, out_dir: str, maneuver_folders: list[str], hyb,
                      dynamics: str = DYNAMICS) -> tuple[list[dict], list[dict]]:
    """Traite tous les maneuver_folders pour un avion donné (plane/root_dir/out_dir).
    Écrit les Excel/PDF habituels dans out_dir. Retourne (all_rows, errors) pour
    permettre à main() de fusionner certains dossiers entre avions."""
    os.makedirs(out_dir, exist_ok=True)
    global_excel = os.path.join(out_dir, "results.xlsx")

    # ── Configure pipelines ───────────────────────────────────────────────────
    config      = AircraftConfig.from_preset(plane, dynamics=dynamics)

    _pipe_kw = dict(
        hyb=hyb, moesp_order=4, SS_f=SS_F,
        aircraft_plane=plane, downsample=DOWNSAMPLE, ds=DS,
    )
    pipe_std = IdentificationPipeline(config, **_pipe_kw)
    pipe_std.data_kw["default_dt"] = 1.0 / FS_DEFAULT
    pipe_std.data_kw["use_accel_lowpass"]   = USE_ACCEL_LOWPASS_RECON
    pipe_std.data_kw["accel_lowpass_fc_hz"] = ACCEL_LOWPASS_FC_HZ

    actuator_coupler = ElevatorActuatorCouplingIdentifier(
        col_col=config.get_col("column"),
        stick_free=STICK_FREE,
    )

    all_rows: list[dict] = []
    errors:   list[dict] = []

    # ── Loop over maneuver folders ────────────────────────────────────────────
    for folder_name in maneuver_folders:
        use_act = USE_ACTUATOR_PIPELINE
        pipe_std.min_trim_dur_s = _min_trim_dur(folder_name)
        pipe_std.maneuver_type  = _maneuver_type(folder_name)
        pipe_std.maneuver_name  = folder_name
        base_dir = os.path.join(root_dir, _resolve_folder_name(folder_name, plane))
        pdf_dir  = os.path.join(out_dir, folder_name)
        os.makedirs(pdf_dir, exist_ok=True)

        safe_name  = (folder_name
                      .replace(" ", "_").replace("-", "_")
                      .replace("(", "").replace(")", ""))
        local_xlsx = os.path.join(out_dir, f"results_{safe_name}.xlsx")

        data_files = sorted(
            glob.glob(os.path.join(base_dir, "**", "*.pkl"), recursive=True)
            + glob.glob(os.path.join(base_dir, "**", "*.pickle"), recursive=True)
            + glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
        )

        if not data_files:
            data_files = sorted(
                glob.glob(os.path.join(base_dir, "*.pkl"))
                + glob.glob(os.path.join(base_dir, "*.pickle"))
                + glob.glob(os.path.join(base_dir, "*.csv"))
            )

        if SINGLE_FILE:
            data_files = [f for f in data_files if os.path.abspath(f) == os.path.abspath(SINGLE_FILE)]

        real_name = _resolve_folder_name(folder_name, plane)
        name_note = f"  (dossier réel : {real_name})" if real_name != folder_name else ""
        print(f"\n{'─'*70}")
        print(f"  Aircraft{plane} / Folder : {folder_name}{name_note}")
        print(f"  Files  : {len(data_files)}")
        print(f"{'─'*70}")

        folder_rows: list[dict] = []

        pbar = tqdm(data_files, unit="file", desc=f"AC{plane} {folder_name[:22]}", dynamic_ncols=True)
        for fp in pbar:
            fname = os.path.basename(fp)
            pbar.set_postfix_str(fname[:40])
            try:
                result = pipe_std.run(fp, verbose=False)
                if use_act:
                    coupling = actuator_coupler.fit(result)
                    if coupling:
                        d = result.metrics.derivatives
                        d["tau_act"]  = coupling["tau"]
                        d["K_act"]    = coupling["K"]
                        d["bw"]       = coupling["bw"]
                        d["bq"]       = coupling["bq"]
                        d["r2_act"]   = coupling["r2"]
                        d["rmse_act"] = coupling["rmse"]
                        d["a_tau_act"]= coupling["a_tau"]
                        d["b_c_act"]  = coupling["b_c"]
                        d["cbw_act"]  = coupling["contrib_bw"]
                        d["cbq_act"]  = coupling["contrib_bq"]
                        result.metrics._eta_e_sim = coupling.get("eta_e_sim")

                row    = _result_to_row(result, folder_name)
                folder_rows.append(row)
                all_rows.append(row)

                # PDF export
                if EXPORT_PDF:
                    pdf_name = os.path.splitext(fname)[0] + ".pdf"
                    pdf_path = os.path.join(pdf_dir, pdf_name)
                    export_pdf(result, pdf_path)

                # AIAA KDE score subfigures
                if EXPORT_AIAA_KDE:
                    stem   = os.path.splitext(fname)[0]
                    aiaa_dir = os.path.join(pdf_dir, "aiaa_kde")
                    save_aiaa_kde_subfigures(result, prefix=stem, out_dir=aiaa_dir)

                # AIAA signals colored by combined KDE score
                if EXPORT_AIAA_COLORED:
                    stem     = os.path.splitext(fname)[0]
                    col_dir  = os.path.join(pdf_dir, "aiaa_colored")
                    save_aiaa_colored_signals(result, prefix=stem, out_dir=col_dir)

                score = result.metrics.global_score
                flag  = "✓" if result.model.is_stable else "⚠"
                coup = _get_coupling(result)
                if coup:
                    tau_v = coup.get("tau_act", float("nan"))
                    K_v   = coup.get("K_act",   float("nan"))
                    bw_v  = coup.get("bw",      float("nan"))
                    bq_v  = coup.get("bq",      float("nan"))
                    r2_v  = coup.get("r2_act",  float("nan"))
                    tqdm.write(
                        f"  {flag}  {fname:<48}  score={score:.3f}  "
                        f"ρ={result.model.spectral_radius:.4f}  "
                        f"τ={tau_v:.3f}s  K={K_v:.3f}  bw={bw_v:.4f}  bq={bq_v:.4f}  R²={r2_v:.3f}"
                    )
                else:
                    tqdm.write(f"  {flag}  {fname:<55}  score={score:.3f}  ρ={result.model.spectral_radius:.4f}")

            except Exception as exc:
                errors.append({"Folder": folder_name, "File": fname, "Error": repr(exc)})
                tqdm.write(f"  ✗  {fname:<55}  {exc}")
                if os.environ.get("DEBUG"):
                    traceback.print_exc()

        # Write per-folder Excel (mirrors local_excel in the notebook cell)
        if folder_rows:
            df_folder = pd.DataFrame(folder_rows)
            df_folder.to_excel(local_xlsx, index=False)
            avg = df_folder["global_score"].mean()
            med = df_folder["global_score"].median()
            print(f"\n  → {local_xlsx}")
            print(f"     avg score={avg:.3f}   median={med:.3f}   n={len(df_folder)}")

    # ── Write global Excel with two sheets (matches notebook output) ──────────
    df_global = pd.DataFrame(all_rows)
    df_errors = pd.DataFrame(errors)

    with pd.ExcelWriter(global_excel, engine="openpyxl") as writer:
        df_global.to_excel(writer, sheet_name="GLOBAL_RESULTS", index=False)
        df_errors.to_excel(writer, sheet_name="ERRORS",         index=False)

    print(f"\n{'='*70}")
    print(f"✅ Aircraft{plane} terminé")
    print(f"   Excel global : {global_excel}")
    print(f"   Manœuvres traitées : {len(all_rows)}   Erreurs : {len(errors)}")

    if not df_global.empty:
        cols_preview = ["File", "global_score", "mach", "alpha_deg", "stable"]
        cols_preview = [c for c in cols_preview if c in df_global.columns]
        print(f"\n{df_global[cols_preview].head(10).to_string(index=False)}")

    return all_rows, errors


def main() -> None:
    # ── Load KDE classifier (partagé entre tous les avions) ───────────────────
    hyb = maneuver_quality_id.TrimDetector.load_hyb(HYB_DIR, which="total", version_tag="")

    combined_rows: dict[str, list[dict]] = {folder: [] for folder in COMBINE_FOLDERS}

    for plane, root_dir, out_dir, maneuver_folders, dynamics in AIRCRAFTS:
        all_rows, _errors = _run_for_aircraft(plane, root_dir, out_dir, maneuver_folders, hyb, dynamics=dynamics)
        for row in all_rows:
            if row.get("Folder") in combined_rows:
                combined_rows[row["Folder"]].append(row)

    # ── Fusionne TOUS les dossiers de COMBINE_FOLDERS dans UN SEUL classeur,
    # une feuille par dossier (un seul fichier Excel, pas un par dossier).
    combined_path = os.path.join(RLS_ROOT, "results_combined.xlsx")
    non_empty = {folder: rows for folder, rows in combined_rows.items() if rows}
    if non_empty:
        with pd.ExcelWriter(combined_path, engine="openpyxl") as writer:
            for folder_name, rows in non_empty.items():
                sheet_name = folder_name[:31]  # limite Excel (31 car. max par feuille)
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"\n  → Fichier combiné : {combined_path}")
        for folder_name, rows in non_empty.items():
            aircraft_tags = sorted({r.get("Aircraft") for r in rows if r.get("Aircraft")})
            print(f"     - {folder_name!r:28s} n={len(rows):4d}  ({', '.join(aircraft_tags)})")


if __name__ == "__main__":
    main()
