#!/usr/bin/env python3
"""
Диагностика демодулированного CVBS (WAV): почему декодер не находит картинку.

По окнам (~2.5 поля, шаг --step) печатает перцентили сигнала, уровни синхро и гашения,
шум на гашении и SNR, число синхроимпульсов, шаг строки в мкс, число длинных импульсов
и вердикт. С --excerpt сохраняет короткий фрагмент записи.

    python3 diagnose_cvbs.py cvbs.wav -s PAL
    python3 diagnose_cvbs.py cvbs.wav -s PAL --step 0.5 --excerpt 3.0 --out excerpt.wav
"""
import argparse
import numpy as np
from scipy.io import wavfile

from decoder import CVBSDecoder
from agc import find_sync_levels
from sync import findpulses, compute_line_len


def analyze(decoder, chunk):
    sp = decoder.sys_params
    fs = decoder.rf.sample_rate
    demod = decoder.rf.demodblock(chunk.astype(np.float64))
    d05, raw = demod["video"]["demod_05"], demod["video"]["demod"]
    q = np.percentile(raw, [1, 50, 99])
    sync_min, blank = find_sync_levels(d05, fs, raw)
    res = dict(p1=q[0], p50=q[1], p99=q[2], sync=sync_min, blank=blank,
               npulses=0, line_us=None, nlong=0, noise=None, snr=None)
    if sync_min is None:
        res["verdict"] = "нет синхро: уровни не найдены"
        return res
    ire = (blank - sync_min) / (-sp["vsync_ire"])
    pulse_max = blank + (sp["vsync_ire"] / 2.0) * ire
    pulses = findpulses(d05, pulse_max, int(fs * 1e-6), int(fs * 35e-6))
    res["npulses"] = len(pulses)
    # шум: СКО на гашении, 3.2-4.2 мкс после конца импульса
    g0, g1 = int(fs * 3.2e-6), int(fs * 4.2e-6)
    segs = [raw[p["start"] + p["len"] + g0:p["start"] + p["len"] + g1] for p in pulses[:300]]
    segs = [s for s in segs if len(s) == g1 - g0]
    if segs:
        res["noise"] = float(np.median([s.std() for s in segs]))
        depth = blank - sync_min
        res["snr"] = depth / max(res["noise"], 1e-12)
    if len(pulses) >= 2:
        res["line_us"] = compute_line_len(pulses) / fs * 1e6
        hs = fs * sp["hsync_pulse_us"] * 1e-6
        res["nlong"] = sum(1 for p in pulses if p["len"] > 3.0 * hs)
    nominal = sp["line_period"]
    if len(pulses) < 50:
        res["verdict"] = "мало синхроимпульсов"
    elif res["line_us"] is None or abs(res["line_us"] - nominal) > 0.03 * nominal:
        res["verdict"] = "шаг строки не тот (шум/не та система/частота)"
    elif res["nlong"] == 0:
        res["verdict"] = "нет кадровых импульсов"
    elif res["snr"] is not None and res["snr"] < 6:
        res["verdict"] = "OK, но очень шумно"
    else:
        res["verdict"] = "OK"
    return res


def estimate_line_rate_ratio(data, fs, system, n_windows=12):
    """
    По синхроимпульсам измеряет:
      1) fH записи к номинальному (соседние импульсы в пределах 2% от строки, усреднение);
      2) амплитуду burst в вольтах: проекция на fsc в окне burst минус проекция в окне без burst.
    Возвращает (ratio, число строк, burst_amp) или (None, 0, None), если синхро мало.
    """
    dec = CVBSDecoder(system, fs / 1e6)
    sp = dec.sys_params
    n = int(dec.inlinelen * sp["field_lines_tbc"] * 2.5)
    nominal = sp["line_period"] * 1e-6 * fs                     # отсчётов на строку
    fsc = sp["fsc_mhz"] * 1e6
    b0, b1 = sp["color_burst_us"]
    diffs, a_burst, a_noise = [], [], []

    def proj(x, i0, i1):                                        # амплитуда на fsc в [i0, i1)
        t = np.arange(i0, i1) / fs
        seg = x[i0:i1] - np.mean(x[i0:i1])
        return 2.0 * np.hypot(np.sum(seg * np.cos(2 * np.pi * fsc * t)),
                              np.sum(seg * np.sin(2 * np.pi * fsc * t))) / (i1 - i0)

    for a in np.linspace(0, max(len(data) - n, 0), n_windows).astype(int):
        chunk = data[a:a + n]
        if len(chunk) < n:
            continue
        demod = dec.rf.demodblock(chunk.astype(np.float64))
        d05, raw = demod["video"]["demod_05"], demod["video"]["demod"]
        sync_min, blank = find_sync_levels(d05, fs, raw)
        if sync_min is None:
            continue
        ire = (blank - sync_min) / (-sp["vsync_ire"])
        pulses = findpulses(d05, blank + (sp["vsync_ire"] / 2.0) * ire, int(fs * 1e-6), int(fs * 35e-6))
        hs = fs * sp["hsync_pulse_us"] * 1e-6
        starts = np.array([p["start"] for p in pulses if p["len"] < 2.0 * hs])
        d = np.diff(starts)
        diffs.extend(d[np.abs(d - nominal) < 0.02 * nominal].tolist())
        wb0, wb1 = int((b0 + 0.3) * 1e-6 * fs), int((b1 - 0.2) * 1e-6 * fs)
        # окно шума той же длины внутри синхроимпульса, там цветности нет
        wn0 = int(1.0e-6 * fs)
        wn1 = wn0 + (wb1 - wb0)
        # берём исходные отсчёты: фильтры декодера ослабляют burst
        src = chunk.astype(np.float64)
        for st in starts:
            if st + max(wb1, wn1) < len(src):
                a_burst.append(proj(src, st + wb0, st + wb1) ** 2)
                a_noise.append(proj(src, st + wn0, st + wn1) ** 2)
    if len(diffs) < 200:
        return None, len(diffs), None
    ratio = float(nominal / np.mean(diffs))
    # разность средних квадратов (burst минус шум) - несмещённая оценка A^2 даже при низком SNR
    amp = float(np.sqrt(max(np.mean(a_burst) - np.mean(a_noise), 0.0)))
    return ratio, len(diffs), amp


def nominal_burst_amplitude(system):
    """Номинальная амплитуда burst, В."""
    from constants import PAL_SYS, NTSC_SYS, VHS_UNDERCOLOR
    sp = PAL_SYS if system.upper() == "PAL" else NTSC_SYS
    return sp["burst_ire"] / 100.0 * VHS_UNDERCOLOR[system.lower()]["volts_per_unit"] / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-s", "--system", choices=["PAL", "NTSC"], default="PAL")
    ap.add_argument("--step", type=float, default=1.0, help="Шаг между окнами, с")
    ap.add_argument("--excerpt", type=float, default=None, help="Сохранить фрагмент с этой секунды")
    ap.add_argument("--excerpt-len", type=float, default=0.1, help="Длина фрагмента, с (0.1 с = 8 МБ)")
    ap.add_argument("--out", default="excerpt.wav")
    args = ap.parse_args()

    fs, data = wavfile.read(args.input)
    if data.ndim > 1:
        data = data[:, 0]
    dec = CVBSDecoder(args.system, fs / 1e6)
    n = int(dec.inlinelen * dec.sys_params["field_lines_tbc"] * 2.5)
    print(f"{args.input}: {len(data) / fs:.2f} с, {fs / 1e6:.3f} МГц, окно {n / fs * 1e3:.0f} мс\n")
    print(f"{'t, с':>6} {'1%':>7} {'50%':>7} {'99%':>7} {'синхро':>7} {'гашение':>8} {'SNR':>5} "
          f"{'импульсов':>9} {'строка, мкс':>11} {'длинных':>7}  вердикт")
    t = 0.0
    while int((t) * fs) + n <= len(data):
        r = analyze(dec, data[int(t * fs):int(t * fs) + n])
        f = lambda v, w, p=3: f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "-"
        print(f"{t:6.2f} {f(r['p1'],7)} {f(r['p50'],7)} {f(r['p99'],7)} {f(r['sync'],7)} {f(r['blank'],8)} "
              f"{f(r['snr'],5,1)} {r['npulses']:9d} {f(r['line_us'],11,2)} {r['nlong']:7d}  {r['verdict']}")
        t += args.step
    ratio, nl, amp = estimate_line_rate_ratio(data, fs, args.system)
    if ratio is None:
        print("\nСтрочная частота: синхро найдено слишком мало для оценки.")
    else:
        nom = nominal_burst_amplitude(args.system)
        print(f"\nСтрочная частота записи / номинал = {ratio:.5f} ({(ratio - 1) * 100:+.3f} %, по {nl} строкам)."
              f"\nBurst: амплитуда {amp:.3f} В при номинале {nom:.3f} В "
              f"(усиление цветности надо умножить на {nom / amp:.2f})." if amp > 0.01 * nom else
              f"\nBurst не обнаружен (ч/б запись или цветность не восстановлена).")
    if args.excerpt is not None:
        a = int(args.excerpt * fs)
        wavfile.write(args.out, int(fs), data[a:a + int(args.excerpt_len * fs)].astype(np.float32))
        print(f"\nФрагмент {args.excerpt}..{args.excerpt + args.excerpt_len} с сохранён: {args.out}")


if __name__ == "__main__":
    main()
