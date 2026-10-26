#!/usr/bin/env python3
"""
Бенчмарк скорости и качества декодера.

Использование:
    python3 benchmark.py                       # PAL+NTSC, 2 секунды, разумные дефолты
    python3 benchmark.py -s PAL --duration 5    # только PAL, 5 секунд
    python3 benchmark.py --quality-only         # пропустить замер скорости
    python3 benchmark.py --json results.json    # сохранить результат в файл
    python3 benchmark.py --markdown             # таблицы в формате Markdown (для README)
    python3 benchmark.py --scopes-dir docs/img  # + осциллограммы/векторограммы
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(cmd, cwd):
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Команда {' '.join(cmd)} упала:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return dt, proc.stdout


from generate_test_signal import DEFAULT_JITTER_NS


def generate_signal(system, duration_s, path, jitter_ns=DEFAULT_JITTER_NS):
    from generate_test_signal import synthesize_cvbs
    from scipy.io import wavfile
    import numpy as np
    sig, fs = synthesize_cvbs(system=system, duration_sec=duration_s, noise_std=0.015, line_jitter_ns=jitter_ns)
    wavfile.write(path, fs, sig.astype(np.float32))
    return fs


def bench_speed(system, wav_path, duration_s, work_dir, comb=False):
    """Возвращает (время этапа 1, время этапа 2, число полей)."""
    nominal_fps_fields = 50.0 if system == "PAL" else (60000.0 / 1001.0)
    frames = int(duration_s * nominal_fps_fields / 2.0 * 1.1) + 2

    base = os.path.join(work_dir, f"bench_{system.lower()}")
    tbc_path = base + ".tbc"
    yuv_path = base + ".yuv"

    dt1, out1 = run([sys.executable, "main.py", os.path.abspath(wav_path), base,
                     "-s", system, "-f", str(frames)], HERE)
    n_fields = None
    # ищем строку "Записано N полей"
    for line in out1.splitlines():
        if "Записано" in line and "полей" in line:
            try:
                n_fields = int(line.split("Записано")[1].split("полей")[0].strip())
            except Exception:
                pass

    dt2, _ = run([sys.executable, "cli.py", os.path.abspath(tbc_path), os.path.abspath(yuv_path),
                     "-s", system] + (["--comb"] if comb else []),
                    os.path.join(HERE, "chroma_decoder"))

    for p in (tbc_path, tbc_path + ".json", yuv_path):
        if os.path.exists(p):
            os.remove(p)

    return dt1, dt2, n_fields


def bench_quality(system, wav_path, duration_s, work_dir, comb=False):
    nominal_fps_fields = 50.0 if system == "PAL" else (60000.0 / 1001.0)
    frames = int(duration_s * nominal_fps_fields / 2.0 * 1.1) + 2
    base = os.path.join(work_dir, f"bench_quality_{system.lower()}")
    tbc_path = base + ".tbc"

    run([sys.executable, "main.py", os.path.abspath(wav_path), base, "-s", system,
        "-f", str(frames)], HERE)
    _, out = run([sys.executable, "_bench_quality.py", os.path.abspath(tbc_path), system]
                + (["--comb"] if comb else []),
                os.path.join(HERE, "chroma_decoder"))
    for p in (tbc_path, tbc_path + ".json"):
        if os.path.exists(p):
            os.remove(p)
    data = json.loads(out.strip().splitlines()[-1])
    return data["scores"]


def fmt_rtf(dt, duration_s):
    return f"{duration_s / dt:5.2f}x" if dt > 0 else "   n/a"


def bench_scopes(system, wav_path, out_dir, work_dir):
    """Прогоняет run_pipeline.py и копирует осциллограмму/векторограмму в out_dir."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    mp4 = os.path.join(work_dir, f"scopes_{system.lower()}.mp4")
    run([sys.executable, "run_pipeline.py", os.path.abspath(wav_path), mp4, "-s", system,
         "-f", "12"], HERE)
    made = []
    for kind in ("cvbs_waveform", "vectorscope"):
        src = os.path.join(work_dir, f"scopes_{system.lower()}_{kind}.png")
        dst = os.path.join(out_dir, f"bench_{system.lower()}_{kind}.png")
        shutil.copy(src, dst)
        made.append(dst)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--system", choices=["PAL", "NTSC", "BOTH"], default="BOTH")
    ap.add_argument("--duration", type=float, default=2.0, help="Длительность тестового сигнала, с")
    ap.add_argument("--jitter-ns", type=float, default=DEFAULT_JITTER_NS,
                    help="Строчный джиттер тестового сигнала, нс [%(default)s]")
    ap.add_argument("--repeats", type=int, default=1, help="Сколько раз повторить замер скорости")
    ap.add_argument("--quality-only", action="store_true")
    ap.add_argument("--speed-only", action="store_true")
    ap.add_argument("--json", default=None, help="Сохранить результаты в JSON-файл")
    ap.add_argument("--markdown", action="store_true", help="Печатать таблицы в формате Markdown")
    ap.add_argument("--scopes-dir", default=None,
                    help="Каталог для осциллограмм и векторограмм (ffmpeg, через run_pipeline.py)")
    ap.add_argument("--work-dir", default="/tmp/radio_decoder_bench")
    args = ap.parse_args()

    import statistics
    os.makedirs(args.work_dir, exist_ok=True)
    systems = ["PAL", "NTSC"] if args.system == "BOTH" else [args.system]
    results = {}
    variants = [("полосовой фильтр", False), ("comb", True)]

    print(f"Генерация тестовых сигналов ({args.duration} с, джиттер {args.jitter_ns} нс)...")
    wavs = {}
    for system in systems:
        path = os.path.join(args.work_dir, f"test_{system.lower()}.wav")
        generate_signal(system, args.duration, path, args.jitter_ns)
        wavs[system] = path
    print()

    def table(headers, rows):
        if args.markdown:
            print("| " + " | ".join(headers) + " |")
            print("|" + "|".join("---" for _ in headers) + "|")
            for r in rows:
                print("| " + " | ".join(str(c) for c in r) + " |")
        else:
            widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
            fmt = "  ".join(f"{{:<{w}}}" for w in widths)
            print(fmt.format(*headers)); print("-" * (sum(widths) + 2 * len(widths)))
            for r in rows:
                print(fmt.format(*[str(c) for c in r]))
        print()

    if not args.quality_only:
        print("=== Производительность ===")
        rows = []
        for system in systems:
            for vname, comb in variants:
                t1s, t2s, n_fields = [], [], None
                for _ in range(args.repeats):
                    dt1, dt2, n_fields = bench_speed(system, wavs[system], args.duration, args.work_dir, comb)
                    t1s.append(dt1); t2s.append(dt2)
                dt1 = sum(t1s) / len(t1s); dt2 = sum(t2s) / len(t2s)
                if not comb:   # этап 1 от Y/C не зависит, печатаем один раз
                    rows.append([system, "-", "CVBS -> TBC", f"{dt1:.2f}", fmt_rtf(dt1, args.duration).strip()])
                rows.append([system, vname, "TBC -> YUV444", f"{dt2:.2f}", fmt_rtf(dt2, args.duration).strip()])
                rows.append([system, vname, "итого", f"{dt1 + dt2:.2f}", fmt_rtf(dt1 + dt2, args.duration).strip()])
                results.setdefault(system, {}).setdefault("speed", {})["comb" if comb else "bandpass"] = dict(
                    cvbs_to_tbc_s=dt1, tbc_to_yuv_s=dt2, n_fields=n_fields,
                    realtime_factor_total=args.duration / (dt1 + dt2))
        table(["система", "Y/C", "этап", "время, с", "real-time фактор"], rows)

    if not args.speed_only:
        print("=== Качество (совпадение цвета с эталоном по 8 цветным полосам) ===")
        rows = []
        for system in systems:
            for vname, comb in variants:
                scores = bench_quality(system, wavs[system], args.duration, args.work_dir, comb)
                n_ok = sum(1 for s_ in scores if s_ > 0)
                rows.append([system, vname, len(scores), f"{min(scores):.3f}",
                             f"{statistics.median(scores):.3f}", f"{max(scores):.3f}",
                             f"{n_ok}/{len(scores)}"])
                results.setdefault(system, {}).setdefault("quality", {})["comb" if comb else "bandpass"] = dict(
                    n_fields_checked=len(scores), min=min(scores), median=statistics.median(scores),
                    max=max(scores), n_correct_sign=n_ok)
        table(["система", "Y/C", "полей", "мин", "медиана", "макс", "верных (>0)"], rows)

    if args.scopes_dir:
        print("=== Осциллограммы и векторограммы (ffmpeg) ===")
        for system in systems:
            for f in bench_scopes(system, wavs[system], args.scopes_dir, args.work_dir):
                print("  " + os.path.relpath(f, HERE))
        print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Результаты сохранены в {args.json}")


if __name__ == "__main__":
    main()
