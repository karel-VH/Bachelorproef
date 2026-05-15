import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from common import (
    SPEED_OF_LIGHT,
    CHANNEL_FREQS,
    N_USED_CHANNELS,
    CHAN_TO_POS,
    DIST_MIN,
    DIST_MAX,
)


def _build_dropout_prior():
    prior = np.ones(N_USED_CHANNELS, dtype=np.float64)

    #2.4 GHz zone
    for ch in range(28, 36):
        if ch in CHAN_TO_POS:
            pos = CHAN_TO_POS[ch]
            for offset in range(-3, 4):
                idx = pos + offset
                if 0 <= idx < N_USED_CHANNELS:
                    prior[idx] += 1.5 * math.exp(-(offset ** 2) / 4.0)

    #bovenband
    for ch in range(58, 68):
        if ch in CHAN_TO_POS:
            pos = CHAN_TO_POS[ch]
            for offset in range(-2, 3):
                idx = pos + offset
                if 0 <= idx < N_USED_CHANNELS:
                    prior[idx] += 0.8 * math.exp(-(offset ** 2) / 4.0)

    #lage kanalen
    for ch in range(5, 15):
        if ch in CHAN_TO_POS:
            prior[CHAN_TO_POS[ch]] += 0.3

    return prior


_DROPOUT_PRIOR = _build_dropout_prior()


def _random_envelope(rng):
    ch = np.arange(N_USED_CHANNELS, dtype=np.float64)
    center = rng.uniform(12.0, 28.0)
    width  = rng.uniform(4.0, 10.0)
    floor  = rng.uniform(0.05, 0.22)

    main_curve = floor + (1.0 - floor) / (1.0 + np.exp((ch - center) / width))
    bumps = gaussian_filter1d(0.06 * rng.standard_normal(N_USED_CHANNELS), sigma=3.0)
    envelope = main_curve * (1.0 + bumps)
    return np.clip(envelope, 0.01, None)


def _sample_dropout_mask(distance, rng):
    base_rate = 0.010 + 0.130 * (distance - DIST_MIN) / (DIST_MAX - DIST_MIN)
    drop_prob = np.clip(base_rate * _DROPOUT_PRIOR, 0.0, 0.80)
    drops = rng.uniform(0.0, 1.0, size=N_USED_CHANNELS) < drop_prob

    if drops.any():
        for i in np.where(drops)[0]:
            for offset in (-1, 1):
                j = i + offset
                if 0 <= j < N_USED_CHANNELS and rng.uniform() < 0.15:
                    drops[j] = True
    return drops


def generate_synthetic_iq(
    distance,
    rng,
    *,
    refl_count_intercept  = 1.5,
    refl_count_slope      = 0.30,
    refl_extra_intercept  = 1.2,
    refl_extra_slope      = 0.20,
    refl_amp_db_mean      = -8.0,
    refl_amp_db_std       = 4.5,
    refl_amp_db_max       = -1.0,
    phase_jitter_min_rad  = 0.10,
    phase_jitter_max_rad  = 0.25,
    snr_db_min_at_zero    = 25.0,
    snr_db_min_slope      = -0.5,
    snr_db_max_at_zero    = 45.0,
    snr_db_max_slope      = -0.6,
    snr_db_floor          = 8.0,
    log_gain_sigma        = 0.45,
):
    f = CHANNEL_FREQS

    #LOS phase ramp
    los_phase = -(4.0 * np.pi * distance * f) / SPEED_OF_LIGHT
    iq = np.exp(1j * los_phase).astype(np.complex128)

    #multipad
    n_refl_mean = refl_count_intercept + refl_count_slope * distance
    n_refl = int(rng.poisson(max(0.0, n_refl_mean)))
    for _ in range(n_refl):
        mean_extra = refl_extra_intercept + refl_extra_slope * distance
        d_extra = min(float(rng.exponential(mean_extra)), 30.0)
        d_refl = distance + d_extra
        amp_db = min(float(rng.normal(refl_amp_db_mean, refl_amp_db_std)), refl_amp_db_max)
        amp = 10.0 ** (amp_db / 20.0)
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        refl_phase = -(4.0 * np.pi * d_refl * f) / SPEED_OF_LIGHT
        iq += amp * np.exp(1j * (refl_phase + phi))

    #phase jitter per kanaal
    pj_std = float(rng.uniform(phase_jitter_min_rad, phase_jitter_max_rad))
    iq *= np.exp(1j * rng.normal(0.0, pj_std, size=N_USED_CHANNELS))

    #antenne envelope
    iq *= _random_envelope(rng)

    #global gain
    iq *= np.exp(rng.normal(0.0, log_gain_sigma))

    #global phase offset
    iq *= np.exp(1j * rng.uniform(0.0, 2.0 * np.pi))

    #additieve ruis met afstandsafhankelijke SNR
    snr_lo = max(snr_db_min_at_zero + snr_db_min_slope * distance, snr_db_floor)
    snr_hi = max(snr_db_max_at_zero + snr_db_max_slope * distance, snr_lo + 1.0)
    snr_db = float(rng.uniform(snr_lo, snr_hi))

    sig_pow = float(np.mean(np.abs(iq) ** 2)) or 1e-12
    noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(N_USED_CHANNELS) + 1j * rng.standard_normal(N_USED_CHANNELS)
    noise *= np.sqrt(noise_pow / 2.0)
    iq += noise

    #channel dropouts
    iq[_sample_dropout_mask(distance, rng)] = 0.0

    return iq.astype(np.complex64)