#!/usr/bin/env python3
"""
Для benchmark.py: качество цвета в .tbc по 8 цветным полосам теста (сумма произведений
медианных U/V на эталонные). Печатает JSON в stdout. Запуск из chroma_decoder/, как cli.py:

    python3 _bench_quality.py path/to.tbc PAL [--comb]
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import get_system, OUTPUT_ZERO, OUT_SCALE
from pal import PALChromaDecoder
from ntsc import NTSCDecoder
from comb import CombFilter2D

COLORS = [[.75, .75, .75], [.75, .75, 0], [0, .75, .75], [0, .75, 0],
          [.75, 0, .75], [.75, 0, 0], [0, 0, .75], [0, 0, 0]]


def score_field(U, V, field_lines):
    active_lines = field_lines - 40
    half = active_lines // 2
    row0 = 20
    r0, r1 = row0 + 20, row0 + half - 20
    col_w = U.shape[1] // 8
    total = 0.0
    for k, c in enumerate(COLORS):
        R, G, B = c
        Yv = 0.299 * R + 0.587 * G + 0.114 * B
        Ue, Ve = 0.493 * (B - Yv), 0.877 * (R - Yv)
        seg = slice(k * col_w + 12, (k + 1) * col_w - 12)
        um, vm = np.median(U[r0:r1, seg]), np.median(V[r0:r1, seg])
        total += um * Ue + vm * Ve
    return total


def main():
    tbc_path, system = sys.argv[1], sys.argv[2]
    use_comb = "--comb" in sys.argv[3:]
    sys_params = get_system(system)
    field_lines = sys_params["field_lines_tbc"]
    fsc = sys_params["fsc_mhz"] * 1e6
    fs = 4 * fsc
    samples_per_line = int(round(fs * sys_params["line_period"] * 1e-6))
    samples_per_field = field_lines * samples_per_line

    tbc_raw = np.fromfile(tbc_path, dtype=np.uint16).astype(np.float64)
    tbc_data = (tbc_raw - OUTPUT_ZERO) / (100.0 * OUT_SCALE)
    n_fields = len(tbc_data) // samples_per_field
    linelocs = np.arange(field_lines + 1) * samples_per_line

    decoder = PALChromaDecoder(fs) if sys_params["chroma"] == "pal" else NTSCDecoder(fs)
    comb = CombFilter2D(fs, fsc) if use_comb else None
    scores = []
    # pal.py печатает отладку в stdout, а benchmark.py читает JSON, поэтому глушим
    quiet = io.StringIO()
    for f in range(n_fields):
        field_data = tbc_data[f * samples_per_field:(f + 1) * samples_per_field]
        comb_chroma = None
        if comb is not None:
            _, comb_chroma = comb.separate_field(field_data.reshape(field_lines, samples_per_line))
        with contextlib.redirect_stdout(quiet):
            if sys_params["chroma"] == "pal":
                Y, U, V = decoder.decode_field(field_data, linelocs, comb_chroma)
            else:
                n = decoder.active_end - decoder.active_start
                Y = np.zeros((field_lines, n)); U = np.zeros((field_lines, n)); V = np.zeros((field_lines, n))
                for i in range(field_lines):
                    s, e = int(linelocs[i]), int(linelocs[i + 1])
                    Y[i], U[i], V[i] = decoder.decode_line(
                        field_data[s:e], None if comb_chroma is None else comb_chroma[i])
        if f < 2:
            continue  # первые поля бывают неполными
        scores.append(score_field(U, V, field_lines))

    sys.stdout.write(json.dumps({"n_fields": n_fields, "scores": scores}))


if __name__ == "__main__":
    main()
