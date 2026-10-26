"""Поиск синхроимпульсов и позиций строк (linelocs)."""
import numpy as np

def findpulses(sync_ref, high, min_synclen=0, max_synclen=5000):
    """Импульсы, где сигнал ниже порога `high`: список {'start': int, 'len': int}."""
    is_low = sync_ref <= high
    diffs = np.diff(is_low.astype(np.int8))

    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1

    if is_low[0]:
        starts = np.insert(starts, 0, 0)
    if is_low[-1]:
        ends = np.append(ends, len(sync_ref))

    lengths = ends - starts
    valid = (lengths >= min_synclen) & (lengths <= max_synclen)

    valid_starts = starts[valid]
    valid_lengths = lengths[valid]

    # импульс с нулевого отсчёта неполный, пропускаем
    mask = valid_starts > 0
    return [{"start": int(s), "len": int(l)} for s, l in zip(valid_starts[mask], valid_lengths[mask])]

def compute_line_len(pulses):
    """Длина строки: медиана расстояний между импульсами."""
    if len(pulses) < 2:
        return 0
    starts = np.array([p["start"] for p in pulses])
    diffs = np.diff(starts)
    return float(np.median(diffs))

def get_line0(pulses, meanlinelen, sys_params):
    """Начало поля (line 0): первый длинный (кадровый) импульс и отступ назад. None, если не нашли."""
    if len(pulses) < 10:
        return None

    hsync_samples = meanlinelen * (sys_params["hsync_pulse_us"] / sys_params["line_period"])

    vsync_locs = []
    for i, p in enumerate(pulses):
        if p["len"] > hsync_samples * 3.0:  # кадровый или уравнительный
            vsync_locs.append(i)

    if not vsync_locs:
        return None

    first_vsync = vsync_locs[0]
    # line 0 примерно за 4 строки до первого длинного импульса
    line0_idx = max(0, first_vsync - 4)
    return pulses[line0_idx]["start"]

def valid_pulses_to_linelocs(pulses, line0loc, meanlinelen, proclines):
    """Позиции строк; на месте пропущенных импульсов - по средней длине строки."""
    linelocs = np.zeros(proclines, dtype=np.float64)

    # импульсы по номерам строк
    pulse_line_map = {}
    for p in pulses:
        line_num = int(np.round((p["start"] - line0loc) / meanlinelen))
        if 0 <= line_num < proclines:
            # из нескольких берём ближайший к идеальной позиции
            ideal_pos = line0loc + line_num * meanlinelen
            if line_num not in pulse_line_map or abs(p["start"] - ideal_pos) < abs(pulse_line_map[line_num]["start"] - ideal_pos):
                pulse_line_map[line_num] = p

    # пропуски заполняем средней длиной строки
    last_valid = line0loc
    for i in range(proclines):
        if i in pulse_line_map:
            linelocs[i] = float(pulse_line_map[i]["start"])
            last_valid = linelocs[i]
        else:
            linelocs[i] = last_valid + meanlinelen
            last_valid = linelocs[i]

    return linelocs
