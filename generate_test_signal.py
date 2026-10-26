"""Синтетический CVBS (PAL/NTSC) для проверки декодера: джиттер, wow/flutter, шум."""
import numpy as np
from scipy.io import wavfile
import scipy.signal as sps
import os

from constants import PAL_SYS, NTSC_SYS

# строчный джиттер по умолчанию, нс (СКО); до ~10 нс декодер работает штатно,
# на 40 нс цвет заметно деградирует
DEFAULT_JITTER_NS = 10.0

def generate_test_pattern(rows, cols):
    """Тестовая картинка: сверху цветные полосы 75%, снизу градации серого. Возвращает Y, U, V."""
    img = np.zeros((rows, cols, 3))

    # полосы (R, G, B)
    colors = [
        [0.75, 0.75, 0.75], # Белый
        [0.75, 0.75, 0.0],  # Желтый
        [0.0, 0.75, 0.75],  # Циан
        [0.0, 0.75, 0.0],   # Зеленый
        [0.75, 0.0, 0.75],  # Пурпурный
        [0.75, 0.0, 0.0],   # Красный
        [0.0, 0.0, 0.75],   # Синий
        [0.0, 0.0, 0.0],    # Черный
    ]

    half_rows = rows // 2
    col_width = cols // 8
    for i, color in enumerate(colors):
        img[:half_rows, i*col_width:(i+1)*col_width] = color

    # градации серого
    for i in range(8):
        val = i * 0.75 / 7.0
        img[half_rows:, i*col_width:(i+1)*col_width] = [val, val, val]

    # RGB -> YUV (BT.601)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = 0.493 * (B - Y)
    V = 0.877 * (R - Y)

    return Y, U, V

def synthesize_cvbs(system="PAL", duration_sec=1.0, fs=None, noise_std=0.015, line_jitter_ns=0.0):
    """Синтетический CVBS с кадровой синхронизацией. Возвращает (сигнал, частота дискретизации)."""
    if system.upper() not in ("PAL", "NTSC"):
        raise ValueError(f"Неизвестная система {system!r}: поддерживаются PAL и NTSC")
    sys_params = PAL_SYS if system.upper() == "PAL" else NTSC_SYS

    if fs is None:
        fs = 4.0 * sys_params["fsc_mhz"] * 1e6

    line_period = sys_params["line_period"] * 1e-6
    fsc = sys_params["fsc_mhz"] * 1e6

    # тайминги, с
    sync_width = 4.7e-6
    vsync_width = 27.3e-6  # широкий импульс (имитация VSYNC)
    # окно burst из constants.py
    burst_start, burst_end = (x * 1e-6 for x in sys_params["color_burst_us"])
    active_start = 10.4e-6
    active_end = sys_params["active_video_us"][1] * 1e-6 if isinstance(sys_params["active_video_us"], (list, tuple)) else 52.0e-6

    # уровни
    sync_level = sys_params["sync_ire"] / 100.0
    setup_level = sys_params["setup_ire"] / 100.0
    # burst_ire в constants.py - размах, амплитуда вдвое меньше
    burst_amp = sys_params["burst_ire"] / 200.0

    # активные строки поля: от 20 до lines_per_field - 20
    active_lines = sys_params["field_lines"][0] - 40
    Y_img, U_img, V_img = generate_test_pattern(rows=active_lines, cols=768)
    _, img_w = Y_img.shape

    N = int(duration_sec * fs)
    t = np.arange(N) / fs

    # wow/flutter
    wow_freq = 0.5
    wow_amp = 1.0e-6
    t_distorted = t + wow_amp * np.sin(2 * np.pi * wow_freq * t)

    # Строчный джиттер - ошибка временной базы, общая для всего сигнала (синхро, burst,
    # видео и поднесущая сдвигаются вместе). Задаётся непрерывной функцией:
    # интерполяция между случайными значениями в серединах строк.
    rng = np.random.default_rng(42)
    n_lines_total = int(t_distorted[-1] / line_period) + 3
    jitter = rng.normal(0, line_jitter_ns * 1e-9, size=n_lines_total)
    centers = (np.arange(n_lines_total) + 0.5) * line_period
    t_distorted = t_distorted + np.interp(t_distorted, centers, jitter)

    line_num = np.floor(t_distorted / line_period).astype(np.int64)
    t_in_line = t_distorted - line_num * line_period

    lines_per_field = sys_params["field_lines"][0]
    line_in_field = line_num % lines_per_field

    # фон - уровень гашения, 0 IRE (7.5 IRE у NTSC только в активной части)
    out = np.zeros(N, dtype=np.float64)

    # синхро: широкие импульсы в строках 0-5 и в последних 5 строках поля
    is_vsync_line = (line_in_field < 6) | (line_in_field >= lines_per_field - 5)
    current_sync_width = np.where(is_vsync_line, vsync_width, sync_width)

    is_sync = (t_in_line >= 0.0) & (t_in_line < current_sync_width)
    out[is_sync] = sync_level

    # burst только на обычных строках
    is_normal_line = (line_in_field >= 20) & (line_in_field < lines_per_field - 20)
    is_burst = (t_in_line >= burst_start) & (t_in_line < burst_end) & is_normal_line

    if system.upper() == "PAL":
        phase = np.where((line_in_field % 2) == 0, 135.0, -135.0) * (np.pi / 180.0)
        v_sign = np.where((line_in_field % 2) == 0, 1.0, -1.0)
    else:
        # NTSC: burst 180°, V-switch нет; массивы нужны для индексации масками
        phase = np.full(N, np.pi)
        v_sign = np.ones(N)

    burst_signal = np.zeros(N)
    # фаза поднесущей считается от непрерывного t_distorted, а не от начала строки
    burst_signal[is_burst] = burst_amp * np.sin(2 * np.pi * fsc * t_distorted[is_burst] + phase[is_burst])
    out += burst_signal

    # активное видео
    is_active = (t_in_line >= active_start) & (t_in_line < active_end) & is_normal_line

    if np.any(is_active):
        x_norm = (t_in_line[is_active] - active_start) / (active_end - active_start)
        col_idx = np.clip((x_norm * (img_w - 1)).astype(np.int32), 0, img_w - 1)
        # строка картинки - позиция внутри активного окна поля
        row_idx = np.clip((line_in_field[is_active] - 20).astype(np.int32), 0, active_lines - 1)

        y_val = Y_img[row_idx, col_idx]
        u_val = U_img[row_idx, col_idx]
        v_val = V_img[row_idx, col_idx]

        carrier_sin = np.sin(2 * np.pi * fsc * t_distorted[is_active])
        carrier_cos = np.cos(2 * np.pi * fsc * t_distorted[is_active])

        if system.upper() == "NTSC":
            # NTSC модулирует I/Q, оси повёрнуты на 33° от U/V. Сам по себе поворот сигнал не
            # меняет, смысл он имеет при разной полосе I и Q (1.3 и 0.5 МГц), поэтому
            # фильтруем их отдельно и построчно: между строками гашение.
            theta = np.deg2rad(33.0)
            nyq_q = fs / 2.0
            i_lpf_b, i_lpf_a = sps.butter(4, min(0.99, 1.3e6 / nyq_q), btype='low')
            q_lpf_b, q_lpf_a = sps.butter(4, min(0.99, 0.5e6 / nyq_q), btype='low')

            chroma = np.zeros_like(u_val)
            active_idx = np.flatnonzero(is_active)
            # границы строк: разрывы в индексах активного видео
            gap_pos = np.flatnonzero(np.diff(active_idx) > 1) + 1
            line_bounds = np.concatenate(([0], gap_pos, [len(active_idx)]))
            starts, ends = line_bounds[:-1], line_bounds[1:]
            lens = ends - starts
            good = lens >= 8  # слишком короткие отрезки не фильтруем
            starts, ends = starts[good], ends[good]
            # все строки одной матрицей и один filtfilt по axis=1: по одной слишком медленно;
            # длина - минимальная, из-за джиттера она почти не меняется.
            # Маска по всему сигналу дорогая, считаем её один раз до цикла
            t_active_all = t_distorted[is_active]
            min_len = int(np.min(ends - starts)) if len(starts) else 0
            if min_len >= 8 and len(starts) > 0:
                n_lines_act = len(starts)
                I_mat = np.empty((n_lines_act, min_len))
                Q_mat = np.empty((n_lines_act, min_len))
                t_mat = np.empty((n_lines_act, min_len))
                for k, (b0, _) in enumerate(zip(starts, ends)):
                    sl = slice(b0, b0 + min_len)
                    I_mat[k] = u_val[sl] * np.cos(theta) - v_val[sl] * np.sin(theta)
                    Q_mat[k] = u_val[sl] * np.sin(theta) + v_val[sl] * np.cos(theta)
                    t_mat[k] = t_active_all[sl]
                I_filt = sps.filtfilt(i_lpf_b, i_lpf_a, I_mat, axis=1)
                Q_filt = sps.filtfilt(q_lpf_b, q_lpf_a, Q_mat, axis=1)
                chroma_mat = (I_filt * np.sin(2 * np.pi * fsc * t_mat - theta)
                              + Q_filt * np.cos(2 * np.pi * fsc * t_mat - theta))
                for k, (b0, b1) in enumerate(zip(starts, ends)):
                    chroma[b0:b0 + min_len] = chroma_mat[k]
                    if b1 > b0 + min_len:  # хвост длиннее min_len без фильтра
                        i_tail = u_val[b0+min_len:b1]*np.cos(theta) - v_val[b0+min_len:b1]*np.sin(theta)
                        q_tail = u_val[b0+min_len:b1]*np.sin(theta) + v_val[b0+min_len:b1]*np.cos(theta)
                        t_tail = t_active_all[b0+min_len:b1]
                        chroma[b0+min_len:b1] = (i_tail*np.sin(2*np.pi*fsc*t_tail-theta)
                                                  + q_tail*np.cos(2*np.pi*fsc*t_tail-theta))
        else:
            chroma = u_val * carrier_sin + (v_val * v_sign[is_active]) * carrier_cos

        vid_level = setup_level + y_val * (1.0 - setup_level) + chroma
        out[is_active] = vid_level

    # шум
    out += rng.normal(0, noise_std, N)

    return out, int(round(fs))

def save_test_files(output_dir="test_signals", duration_sec=1.0, jitter_ns=DEFAULT_JITTER_NS):
    """Генерирует и сохраняет тестовые файлы для PAL и NTSC (float32 WAV)."""
    os.makedirs(output_dir, exist_ok=True)
    for system in ("PAL", "NTSC"):
        print(f"Генерация {system} CVBS ({duration_sec} с, джиттер {jitter_ns} нс)...")
        sig, fs = synthesize_cvbs(system=system, duration_sec=duration_sec, noise_std=0.015,
                                  line_jitter_ns=jitter_ns)
        path = os.path.join(output_dir, f"test_{system.lower()}_cvbs.wav")
        # float32, чтобы не было клиппинга
        wavfile.write(path, fs, sig.astype(np.float32))
        print(f"Сохранено: {path} ({len(sig)} отсчётов, {fs / 1e6:.3f} МГц)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Генератор синтетического CVBS (PAL и NTSC) для тестов декодера")
    ap.add_argument("-o", "--output-dir", default="test_signals")
    ap.add_argument("-d", "--duration", type=float, default=1.0, help="Длительность, с")
    ap.add_argument("--jitter-ns", type=float, default=DEFAULT_JITTER_NS, help="Строчный джиттер, нс")
    a = ap.parse_args()
    save_test_files(a.output_dir, a.duration, a.jitter_ns)
    print("Готово.")
