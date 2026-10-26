#!/usr/bin/env python3
"""
Эмулятор ВЧ-сигнала с головок VHS (color-under) для проверки fm_demod.py --undercolor.

Берёт синтетический CVBS, переносит цветность вниз (~627/629 кГц), яркостью модулирует
ЧМ-несущую (3.8 МГц PAL / 3.4 МГц NTSC на вершине синхро): ВЧ = ЧМ + k * color-under + шум.
Выход - 2-канальный float32 WAV (канал 0 - ВЧ, канал 1 - нули), 20 МГц по умолчанию.
Не эмулируются построчный поворот фазы цветности, джиттер ленты и дропауты.

    python3 generate_undercolor_signal.py -s PAL -d 0.5 -o test_signals/test_pal_undercolor_rf.wav
"""
import argparse
import os

import numpy as np
import scipy.signal as sps
from scipy.io import wavfile

from constants import VHS_UNDERCOLOR, FM_HZ_PER_VOLT
from generate_test_signal import synthesize_cvbs

UNDER_LEVEL = 0.1   # амплитуда color-under относительно амплитуды ЧМ-несущей (1.0)


def record_undercolor_rf(system="PAL", duration_sec=0.5, fs=20e6, noise_std=0.01,
                         line_jitter_ns=0.0, under_level=UNDER_LEVEL):
    """Возвращает (rf, fs): вещественный ВЧ-сигнал и частота дискретизации."""
    p = VHS_UNDERCOLOR[system.lower()]
    cvbs, fs_int = synthesize_cvbs(system=system, duration_sec=duration_sec, fs=fs,
                                   noise_std=0.0, line_jitter_ns=line_jitter_ns)
    fs = float(fs_int)
    v = cvbs * p["volts_per_unit"]                  # в вольтах: синхро ~ -0.3, белый ~ +0.7
    nyq = fs / 2.0
    t = np.arange(len(v)) / fs

    # цветность (полоса вокруг fsc) и яркость
    b, a = sps.butter(3, [(p["fsc"] - 1.2e6) / nyq, (p["fsc"] + 1.2e6) / nyq], btype="band")
    chroma = sps.filtfilt(b, a, v)
    bl, al = sps.butter(4, 3.2e6 / nyq, btype="low")
    luma = sps.filtfilt(bl, al, v - chroma)

    # перенос цветности вниз гетеродином fsc + f_under
    lo = np.cos(2 * np.pi * (p["fsc"] + p["under_freq"]) * t)
    bu, au = sps.butter(4, 1.3e6 / nyq, btype="low")
    under = sps.filtfilt(bu, au, 2.0 * chroma * lo)

    # ЧМ яркости: f = f_black + 1 МГц/В * v
    f_black = p["fm_sync"] + p["sync_frac"] * FM_HZ_PER_VOLT
    f_inst = f_black + luma * FM_HZ_PER_VOLT
    fm = np.cos(2 * np.pi * np.cumsum(f_inst) / fs)

    rf = fm + under_level * under
    rf += np.random.default_rng(1).normal(0.0, noise_std, len(rf))
    return rf, fs


def main():
    ap = argparse.ArgumentParser(description="Эмулятор ВЧ-сигнала color-under (VHS) для fm_demod.py --undercolor")
    ap.add_argument("-s", "--system", choices=["PAL", "NTSC"], default="PAL")
    ap.add_argument("-d", "--duration", type=float, default=0.5, help="Длительность, с")
    ap.add_argument("--samp-rate", type=float, default=20e6, help="Частота дискретизации, Гц")
    ap.add_argument("--noise", type=float, default=0.01, help="СКО шума (ЧМ-несущая = 1.0)")
    ap.add_argument("--jitter-ns", type=float, default=0.0, help="Строчный джиттер, нс")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()
    out = args.output or os.path.join("test_signals", f"test_{args.system.lower()}_undercolor_rf.wav")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    rf, fs = record_undercolor_rf(args.system, args.duration, args.samp_rate, args.noise, args.jitter_ns)
    wavfile.write(out, int(fs), np.stack([rf, np.zeros_like(rf)], axis=1).astype(np.float32))
    print(f"Сохранено: {out} ({len(rf)} отсчётов, {fs / 1e6:.1f} МГц, 2 канала: ВЧ + нули)")


if __name__ == "__main__":
    main()
