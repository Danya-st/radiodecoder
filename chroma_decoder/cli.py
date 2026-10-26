"""Хрома-декодер: .tbc -> rawvideo (yuv444p)."""
import argparse

import numpy as np

from ntsc import NTSCDecoder
from pal import PALChromaDecoder
from comb import CombFilter2D

OUTPUT_ZERO = 16384.0
OUT_SCALE = 327.68

# декодер отдаёт U = 0.493*(B-Y), V = 0.877*(R-Y); для Cb/Cr BT.601 в 8 бит
# нужны свои множители
U_TO_CB = 224.0 / (0.493 * 1.772)
V_TO_CR = 224.0 / (0.877 * 1.402)


def decode_tbc_to_raw(tbc_path, system, output_path, use_comb=False):
    """Декодирует .tbc в планарный YUV444, возвращает (строк в поле, отсчётов в активной части)."""
    # 100 IRE = 1.0
    tbc_raw = np.fromfile(tbc_path, dtype=np.uint16).astype(np.float64)
    tbc_data = (tbc_raw - OUTPUT_ZERO) / (100.0 * OUT_SCALE)

    if system.upper() == "PAL":
        field_lines = 313
        line_period_us = 64.0
        fsc = 4.43361875e6
        sample_rate = 4 * fsc
        DecoderClass = PALChromaDecoder
    elif system.upper() == "NTSC":
        field_lines = 263
        line_period_us = 63.555
        fsc = 3.579545e6
        sample_rate = 4 * fsc
        DecoderClass = NTSCDecoder
    else:
        raise ValueError(f"Неизвестная система: {system}")

    samples_per_line = int(round(sample_rate * line_period_us * 1e-6))

    samples_per_field = field_lines * samples_per_line
    decoder = DecoderClass(sample_rate)
    comb = CombFilter2D(sample_rate, fsc) if use_comb else None
    if system.upper() == "NTSC":
        active_samples = decoder.active_end - decoder.active_start
    else:
        active_samples = int(sample_rate * 52.0e-6)

    with open(output_path, "wb") as out_f:
        field_idx = 0
        offset = 0

        while offset + samples_per_field <= len(tbc_data):
            field_data = tbc_data[offset : offset + samples_per_field]
            linelocs = np.arange(field_lines + 1) * samples_per_line

            # гребёнка: цветность всего поля сразу
            comb_chroma = None
            if comb is not None:
                _, comb_chroma = comb.separate_field(field_data.reshape(field_lines, samples_per_line))

            # PAL: поле целиком (нужно для чётности поля и PAL delay)
            if isinstance(decoder, PALChromaDecoder):
                Y, U, V = decoder.decode_field(field_data, linelocs, comb_chroma)
            else:
                # NTSC: построчно
                n_lines = field_lines
                for i in range(n_lines):
                    start = int(linelocs[i])
                    end = int(linelocs[i + 1])
                    line = field_data[start:end]

                    y_full, u_full, v_full = decoder.decode_line(
                        line, None if comb_chroma is None else comb_chroma[i])

                    # размер по первой строке
                    if i == 0:
                        active_samples = len(y_full)
                        Y = np.zeros((n_lines, active_samples))
                        U = np.zeros((n_lines, active_samples))
                        V = np.zeros((n_lines, active_samples))

                    # строка у края файла может быть короче
                    n = min(len(y_full), active_samples)
                    Y[i, :n] = y_full[:n]
                    U[i, :n] = u_full[:n]
                    V[i, :n] = v_full[:n]

            # в 8 бит
            Y_u8 = np.clip(Y * 219 + 16, 16, 235).astype(np.uint8)
            U_u8 = np.clip(U * U_TO_CB + 128, 16, 240).astype(np.uint8)
            V_u8 = np.clip(V * V_TO_CR + 128, 16, 240).astype(np.uint8)

            # планарный YUV444
            out_f.write(Y_u8.tobytes())
            out_f.write(U_u8.tobytes())
            out_f.write(V_u8.tobytes())

            field_idx += 1
            offset += samples_per_field
            print(f"\rДекодировано полей: {field_idx}", end="")

    print(f"\nГотово. Записано {field_idx} полей в {output_path}")
    return field_lines, active_samples


def main():
    parser = argparse.ArgumentParser(
        description="Хрома-декодер TBC -> rawvideo"
    )
    parser.add_argument("input", help="Входной .tbc файл")
    parser.add_argument("output", help="Выходной .yuv файл")
    parser.add_argument(
        "-s", "--system", choices=["PAL", "NTSC"], default="PAL"
    )
    parser.add_argument(
        "--comb", action="store_true",
        help="Выделять цветность гребенчатым фильтром (comb.py; NTSC: строки ±1, PAL: ±2) "
             "вместо одного лишь полосового фильтра"
    )
    args = parser.parse_args()

    decode_tbc_to_raw(args.input, args.system, args.output, use_comb=args.comb)


if __name__ == "__main__":
    main()
