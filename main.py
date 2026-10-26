"""CLI: CVBS -> .tbc."""
import argparse
import os
import sys
from decoder import CVBSDecoder
from inout import read_input, write_tbc

def main():
    parser = argparse.ArgumentParser(description="Декодер CVBS (PAL/NTSC) в TBC")
    parser.add_argument("input", help="Входной файл (.r16, .s16, .wav)")
    parser.add_argument("output", help="Базовое имя выходного файла (будет создан .tbc)")
    parser.add_argument("-s", "--system", choices=["PAL", "NTSC"], default="PAL", help="Видеосистема")
    parser.add_argument("-r", "--sample-rate", type=float, default=None, help="Частота дискретизации входа в МГц (по умолчанию: 4 * fsc)")
    parser.add_argument("-f", "--frames", type=int, default=10, help="Количество кадров для декодирования")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Ошибка: Входной файл '{args.input}' не найден.")
        sys.exit(1)

    print(f"Чтение {args.input}...")
    data, sample_rate_hz = read_input(args.input)

    sample_rate_mhz = args.sample_rate or (sample_rate_hz / 1e6)

    print(f"Инициализация декодера для {args.system} при {sample_rate_mhz:.3f} МГц...")
    decoder = CVBSDecoder(system=args.system, sample_rate_mhz=sample_rate_mhz)

    out_tbc = args.output + ".tbc"
    if os.path.exists(out_tbc):
        os.remove(out_tbc)

    field_lines = decoder.sys_params["field_lines_tbc"]
    samples_per_field = int(decoder.inlinelen * field_lines)
    overlap = int(1.5 * samples_per_field)

    print(f"Декодирование {args.frames} кадров ({args.frames * 2} полей)...")
    fields_decoded = 0
    target_fields = args.frames * 2
    offset = 0

    while fields_decoded < target_fields and offset < len(data):
        chunk = data[offset : offset + samples_per_field + overlap]
        if len(chunk) < samples_per_field:
            break

        field = decoder.decode_field(chunk)
        if field and field.valid:
            tbc_data = field.scale_to_16bit()
            write_tbc(out_tbc, tbc_data)
            fields_decoded += 1
            print(f"\rДекодировано полей: {fields_decoded}/{target_fields}", end="")

            measured_len = field.linelocs[field.outlinecount] - field.linelocs[0]
            offset += max(1, int(round(measured_len)))
        else:
            offset += max(1, samples_per_field // 2)

    print(f"\nГотово. Записано {fields_decoded} полей в {out_tbc}")


if __name__ == "__main__":
    main()
