#!/usr/bin/env python3
"""
Весь конвейер: CVBS (.wav) -> .tbc -> .yuv -> .mp4, плюс осциллограмма и векторограмма.
С --from-iq сначала запускается fm_demod.py (нужен gnuradio, в requirements.txt его нет).

    python3 run_pipeline.py capture.wav out.mp4 -s PAL
    python3 run_pipeline.py capture.wav out.mp4 -s NTSC -r 14.318182 --comb
    python3 run_pipeline.py iq.wav out.mp4 -s PAL --from-iq --fm-deviation 4.5
    python3 run_pipeline.py head_rf.wav out.mp4 -s NTSC --from-iq --undercolor -r 14.318182
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))

# геометрия сырого YUV и параметры ffmpeg; кроп убирает затемнённые края кадра
PIPELINE_PARAMS = {
    "PAL": dict(
        tbc_line_len=1135,               # отсчётов на строку в .tbc (сетка 4*fsc)
        frame_lines=313,                 # строк в поле
        active_width=951,                # отсчётов в строке
        in_fps=50,                       # полей в секунду
        out_fps=25,
        crop="920:288:0:12",             # w:h:x:y
        scale="720:576",
    ),
    "NTSC": dict(
        tbc_line_len=910,
        frame_lines=263,
        active_width=744,
        in_fps=60,
        out_fps="30000/1001",            # 29.97
        crop="720:230:16:21",
        scale="720:480",
    ),
}

DEINTERLACE_FILTERS = {
    "PAL": {"yadif": "yadif=0:-1:0", "w3fdif": "w3fdif", "none": None},
    "NTSC": {"yadif": "yadif=0:-1:0", "w3fdif": "w3fdif", "none": None},
}

FM_DEMOD_SCRIPT = os.path.join(HERE, "fm_demod.py")


def wav_info(path):
    """Частота дискретизации (Гц) и число отсчётов, данные в память не грузим."""
    from scipy.io import wavfile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate, data = wavfile.read(path, mmap=True)
    return rate, len(data)


def estimate_frames(n_samples, sample_rate_mhz, system, margin=1.15):
    """Число кадров по длительности файла с запасом margin; main.py остановится сам, когда данные кончатся."""
    duration_s = n_samples / (sample_rate_mhz * 1e6)
    fps_fields = 50.0 if system.upper() == "PAL" else (60000.0 / 1001.0)
    return max(int(duration_s * fps_fields / 2.0 * margin) + 2, 2)


# фильтр vectorscope: U/V кадра на круге BT.601
VECTORSCOPE_FILTER = ("vectorscope=mode=color2:colorspace=601:graticule=green:"
                      "flags=name:envelope=instant:intensity=0.05")


def pick_scope_frame(n_frames, requested=None):
    """Кадр для снимка: заданный или середина файла (первые кадры бывают неполными)."""
    if requested is not None:
        return max(0, requested)
    return max(0, min(n_frames - 1, max(2, n_frames // 2)))


def cvbs_waveform_chain(height, width=None):
    """
    Цепочка ffmpeg: 16-битное серое видео из .tbc -> осциллограмма CVBS со шкалой IRE.
    В 8 битах 0 IRE = 64, 100 IRE = 192, -40 IRE ~ 13. Шкалу рисуем сами: встроенная
    у waveform рассчитана на диапазон 16..235. height/width - размер картинки.
    """
    def y_of(ire):
        return int(round((255 - (64 + 1.28 * ire)) * height / 256.0))
    marks = [(-40, "-40 IRE"), (0, "0 IRE"), (100, "100 IRE")]
    grid = ",".join(f"drawbox=y={y_of(i)}:w=iw:h=1:color=green" for i, _ in marks)
    text = ",".join(f"drawtext=text='{t}':x=4:y={y_of(i) - 16 if i >= 0 else y_of(i) + 2}:"
                    f"fontcolor=green:fontsize=14" for i, t in marks)
    return (f"format=gray,waveform=mode=column:components=1:filter=lowpass:"
            f"scale=digital:graticule=none:intensity=0.1:display=overlay,"
            f"scale={width or 'iw'}:{height}:flags=neighbor,{grid},{text}")


def tbc_input_args(tbc):
    """Аргументы ffmpeg для чтения .tbc как серого 16-битного видео (поле = кадр)."""
    return ["-f", "rawvideo", "-pixel_format", "gray16le",
            "-video_size", f"{tbc['line_len']}x{tbc['field_lines']}",
            "-framerate", str(tbc["in_fps"]), "-i", os.path.abspath(tbc["path"])]


def render_cvbs_waveform(tbc, out_path, field):
    """Осциллограмма CVBS по одному полю из .tbc (PNG)."""
    vf = f"select=eq(n\\,{field}),{cvbs_waveform_chain(512)}"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + tbc_input_args(tbc) + [
        "-vf", vf, "-frames:v", "1", os.path.abspath(out_path)]
    run_stage(cmd, HERE, "4/4 scopes -> " + os.path.basename(out_path))
    return out_path


def render_scopes(mp4_path, out_base, out_height, frame, tbc, video=False):
    """
    Векторограмма кадра из .mp4 (PNG); с video=True ещё <out_base>_scopes.mp4:
    картинка | осциллограмма CVBS | векторограмма. tbc - dict с параметрами .tbc.
    Возвращает список созданных файлов.
    """
    made = []
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", os.path.abspath(mp4_path)]
    path = out_base + "_vectorscope.png"
    vf = f"select=eq(n\\,{frame}),{VECTORSCOPE_FILTER},scale=768:768:flags=neighbor"
    run_stage(base + ["-vf", vf, "-frames:v", "1", os.path.abspath(path)], HERE,
              "4/4 scopes -> " + os.path.basename(path))
    made.append(path)
    if video:
        path = out_base + "_scopes.mp4"
        vec = f"{VECTORSCOPE_FILTER},scale={out_height}:{out_height}:flags=neighbor"
        # вход 1 - .tbc, приводим к частоте кадров видео
        wf = f"[1:v]fps={tbc['out_fps']},{cvbs_waveform_chain(out_height, 720)}[w]"
        fc = f"[0:v]split=2[a][c];{wf};[c]{vec}[v];[a][w][v]hstack=inputs=3:shortest=1,format=yuv420p"
        cmd = base + tbc_input_args(tbc)
        run_stage(cmd + ["-filter_complex", fc, "-c:v", "libx264", "-crf", "20",
                         os.path.abspath(path)], HERE, "4/4 scopes -> " + os.path.basename(path))
        made.append(path)
    return made


def run_stage(cmd, cwd, label):
    print(f"\n[{label}] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    dt = time.time() - t0
    for line in (proc.stdout.strip().splitlines() or [""])[-3:]:
        print("  " + line)
    if proc.returncode != 0:
        print(f"[{label}] ОШИБКА (код {proc.returncode}). Полный вывод:")
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"Этап '{label}' завершился с ошибкой")
    print(f"[{label}] готово за {dt:.2f} с")
    return dt


def run_fm_demod(iq_wav, cvbs_wav, sample_rate_mhz, deviation_mhz, video_bw_mhz,
                 undercolor=False, system="PAL"):
    """Этап 0: I/Q или ВЧ -> CVBS через fm_demod.py (headless). Нужны gnuradio и PyQt5."""
    if not os.path.exists(FM_DEMOD_SCRIPT):
        raise RuntimeError(f"Не найден {FM_DEMOD_SCRIPT}")
    cmd = [
        sys.executable, FM_DEMOD_SCRIPT, "--headless",
        "-i", os.path.abspath(iq_wav), "-o", os.path.abspath(cvbs_wav),
        "--samp-rate", str(int(round(sample_rate_mhz * 1e6))),
        "--deviation", str(deviation_mhz * 1e6),
        "--video-bw", str(video_bw_mhz * 1e6),
        "--system", system.lower(),
    ]
    if undercolor:
        cmd.append("--undercolor")
    print(f"\n[0/4 I/Q -> CVBS (fm_demod)] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        combined = proc.stdout + proc.stderr
        if "ModuleNotFoundError: No module named 'gnuradio'" in combined \
                or "ModuleNotFoundError: No module named 'PyQt5'" in combined:
            raise RuntimeError(
                "--from-iq требует gnuradio и PyQt5, их нет в requirements.txt. Установите "
                "GNU Radio (см. README) или продемодулируйте запись заранее и запустите "
                "run_pipeline.py без --from-iq."
            )
        print(f"[0/4 I/Q -> CVBS (fm_demod)] ОШИБКА (код {proc.returncode}):")
        print(combined[-3000:])
        raise RuntimeError("Этап '0/4 I/Q -> CVBS (fm_demod)' завершился с ошибкой")
    for line in (proc.stdout.strip().splitlines() or [""])[-3:]:
        print("  " + line)
    print(f"[0/4 I/Q -> CVBS (fm_demod)] готово за {dt:.2f} с")
    return dt


def run_pipeline(input_wav, output_mp4, system, frames=None, sample_rate_mhz=None,
                 deinterlace="yadif", from_iq=False, fm_deviation_mhz=5.0, fm_video_bw_mhz=6.0,
                 undercolor=False, comb=False, scopes=True, scopes_video=False,
                 scope_frame=None):
    system = system.upper()
    if system not in PIPELINE_PARAMS:
        raise ValueError(f"run_pipeline.py поддерживает только PAL и NTSC, получено: {system}")
    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Входной файл не найден: {input_wav}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg не найден в PATH. Установите ffmpeg, чтобы получить .mp4.")
    if deinterlace not in DEINTERLACE_FILTERS[system]:
        raise ValueError(f"--deinterlace должен быть одним из {list(DEINTERLACE_FILTERS[system])}")

    params = PIPELINE_PARAMS[system]
    # частота входа: из -r, иначе из заголовка WAV
    in_rate_hz, n_samples = wav_info(input_wav)
    rate_mhz = sample_rate_mhz or in_rate_hz / 1e6
    print(f"Частота дискретизации входа: {rate_mhz:.6f} МГц"
          f"{'' if sample_rate_mhz else ' (из заголовка WAV)'}")
    if frames is None:
        frames = estimate_frames(n_samples, rate_mhz, system)
        print(f"Число кадров не задано - оценка по длительности файла: {frames}")

    base = os.path.splitext(os.path.basename(input_wav))[0]
    timings = {}
    with tempfile.TemporaryDirectory(prefix="radio_decoder_") as work:
        tbc_base = os.path.join(work, base)
        tbc_path = tbc_base + ".tbc"
        yuv_path = os.path.join(work, base + ".yuv")

        # этап 0 (опционально): сырая запись -> CVBS
        if from_iq:
            cvbs_path = os.path.join(work, base + "_cvbs.wav")
            timings["iq_to_cvbs"] = run_fm_demod(input_wav, cvbs_path, rate_mhz, fm_deviation_mhz,
                                                 fm_video_bw_mhz, undercolor, system)
            if not os.path.exists(cvbs_path):
                raise RuntimeError(f"fm_demod.py не создал ожидаемый файл {cvbs_path}")
            input_wav = cvbs_path   # дальше как обычный CVBS

        # этап 1: CVBS -> TBC
        cmd1 = [sys.executable, "main.py", os.path.abspath(input_wav), tbc_base, "-s", system,
                "-f", str(frames), "-r", str(rate_mhz)]
        timings["cvbs_to_tbc"] = run_stage(cmd1, HERE, "1/4 CVBS -> TBC")
        if not os.path.exists(tbc_path):
            raise RuntimeError(f"main.py не создал ожидаемый файл {tbc_path}")

        # этап 2: TBC -> YUV444
        cmd2 = [sys.executable, "cli.py", os.path.abspath(tbc_path), os.path.abspath(yuv_path),
                "-s", system] + (["--comb"] if comb else [])
        timings["tbc_to_yuv"] = run_stage(cmd2, os.path.join(HERE, "chroma_decoder"),
                                          "2/4 TBC -> YUV444")
        if not os.path.exists(yuv_path):
            raise RuntimeError(f"chroma_decoder/cli.py не создал ожидаемый файл {yuv_path}")

        # этап 3: YUV444 -> MP4
        vf_parts = [f"crop={params['crop']}"]
        deint_filter = DEINTERLACE_FILTERS[system][deinterlace]
        if deint_filter:
            vf_parts.append(deint_filter)
        vf_parts.append(f"scale={params['scale']}")

        os.makedirs(os.path.dirname(os.path.abspath(output_mp4)) or ".", exist_ok=True)
        cmd3 = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "yuv444p",
            "-video_size", f"{params['active_width']}x{params['frame_lines']}",
            "-framerate", str(params["in_fps"]),
            "-i", os.path.abspath(yuv_path),
            "-vf", ",".join(vf_parts),
            "-r", str(params["out_fps"]),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            os.path.abspath(output_mp4),
        ]
        timings["yuv_to_mp4"] = run_stage(cmd3, HERE, "3/4 YUV444 -> MP4")

        # этап 4: осциллограмма и векторограмма
        scope_files = []
        if scopes:
            n_frames = os.path.getsize(yuv_path) // (3 * params["active_width"] * params["frame_lines"]) // 2
            out_base = os.path.splitext(os.path.abspath(output_mp4))[0]
            ts = time.time()
            out_h = int(params["scale"].split(":")[1])
            tbc_info = dict(path=tbc_path, line_len=params["tbc_line_len"],
                            field_lines=params["frame_lines"], in_fps=params["in_fps"],
                            out_fps=params["out_fps"])
            scope_files = [render_cvbs_waveform(
                tbc_info, out_base + "_cvbs_waveform.png", pick_scope_frame(n_frames * 2))]
            scope_files += render_scopes(
                output_mp4, out_base, out_h, pick_scope_frame(n_frames, scope_frame),
                tbc_info, video=scopes_video)
            timings["scopes"] = time.time() - ts

    size_mb = os.path.getsize(output_mp4) / 1e6
    print(f"\nГотово: {output_mp4} ({size_mb:.1f} МБ)")
    for sf in scope_files:
        print(f"Скоп: {sf}")
    stage0_s = f"этап0={timings['iq_to_cvbs']:.2f}с  " if "iq_to_cvbs" in timings else ""
    stage4_s = f"  этап4={timings['scopes']:.2f}с" if "scopes" in timings else ""
    print(f"Время: {stage0_s}этап1={timings['cvbs_to_tbc']:.2f}с  этап2={timings['tbc_to_yuv']:.2f}с  "
          f"этап3={timings['yuv_to_mp4']:.2f}с{stage4_s}  итого={sum(timings.values()):.2f}с")
    return timings


def main():
    ap = argparse.ArgumentParser(
        description="Полный пайплайн: CVBS (.wav) -> TBC -> YUV444 -> MP4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="Входной .wav (CVBS, либо сырая запись с --from-iq)")
    ap.add_argument("output", help="Выходной .mp4")
    ap.add_argument("-s", "--system", choices=["PAL", "NTSC"], default="PAL")
    ap.add_argument("-f", "--frames", type=int, default=None,
                    help="Сколько кадров декодировать (по умолчанию - весь файл)")
    ap.add_argument("-r", "--sample-rate", type=float, default=None,
                    help="Реальная частота дискретизации входа, МГц (по умолчанию - из "
                         "заголовка WAV; если он неверен, задайте точно, напр. 14.318182)")
    ap.add_argument("--deinterlace", choices=["yadif", "w3fdif", "none"], default="yadif")
    ap.add_argument("--from-iq", action="store_true",
                    help="Вход - сырая I/Q- или ВЧ-запись (2-канальный WAV), а не CVBS; "
                         "сначала запускается fm_demod.py (нужен gnuradio, см. README)")
    ap.add_argument("--fm-deviation", type=float, default=5.0,
                    help="[--from-iq] Девиация ЧМ несущей, МГц")
    ap.add_argument("--fm-video-bw", type=float, default=6.0,
                    help="[--from-iq] Полоса ФНЧ после демодулятора, МГц")
    ap.add_argument("--undercolor", action="store_true",
                    help="[--from-iq] Цветность под ЧМ-яркостью (color-under, VHS): вход - "
                         "ВЧ-сигнал с головок в канале 0 WAV")
    ap.add_argument("--comb", action="store_true",
                    help="Разделять Y/C гребенчатым фильтром (chroma_decoder/comb.py)")
    ap.add_argument("--no-scopes", action="store_true",
                    help="Не строить осциллограмму и векторограмму (.png рядом с .mp4)")
    ap.add_argument("--scopes-video", action="store_true",
                    help="Дополнительно собрать <имя>_scopes.mp4: картинка | осциллограмма | векторограмма")
    ap.add_argument("--scope-frame", type=int, default=None,
                    help="Номер кадра для .png (по умолчанию - середина файла)")
    args = ap.parse_args()

    try:
        run_pipeline(args.input, args.output, args.system, args.frames, args.sample_rate,
                     args.deinterlace, args.from_iq, args.fm_deviation, args.fm_video_bw,
                     args.undercolor, args.comb, not args.no_scopes, args.scopes_video,
                     args.scope_frame)
    except Exception as e:
        print(f"\nОшибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
