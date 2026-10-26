"""Уровни синхро и гашения, масштабирование поля в 16 бит."""
import numpy as np

def find_sync_levels(demod_05, sample_rate_hz, demod_raw=None):
    """
    Уровни синхро и гашения (back porch).
    Позиции импульсов ищем по demod_05 (после ФНЧ 0.5 МГц), уровни берём медианой из
    demod_raw: после filtfilt на краях синхроимпульсов остаётся звон, он искажает вершину.
    """
    data = demod_05[int(sample_rate_hz * 1e-6):]  # 1 мкс с краю отбрасываем
    if len(data) == 0:
        return None, None

    sync_min, max_val = np.percentile(data, [3, 99])
    difference = max_val - sync_min

    if difference == 0:
        return None, None

    on_sync = data < (sync_min + (difference / 15.0))

    offset = 0
    blank_level = None
    found_porch = False

    while not found_porch and offset < len(data) - 100:
        sync_indices = np.where(on_sync[offset:])[0]
        if len(sync_indices) == 0:
            offset += int(sample_rate_hz * 50e-6)
            continue

        search_start = sync_indices[0] + offset
        next_cross = np.where(~on_sync[search_start:])[0]
        if len(next_cross) == 0:
            break

        next_cross = next_cross[0] + search_start
        # back porch: 4 мкс после конца синхро
        porch_sample = next_cross + int(sample_rate_hz * 4.0e-6)

        if porch_sample < len(data):
            blank_level = data[porch_sample]
            if blank_level > sync_min + (difference / 15.0):
                found_porch = True
            else:
                offset += int(sample_rate_hz * 50e-6)
        else:
            break

    if found_porch and demod_raw is not None:
        raw = demod_raw[int(sample_rate_hz * 1e-6):]
        # вершина синхро по несглаженному сигналу
        sync_min = float(np.median(raw[on_sync]))
        # гашение: 3.2-4.2 мкс после конца синхро, по первым 200 импульсам
        gate0, gate1 = int(sample_rate_hz * 3.2e-6), int(sample_rate_hz * 4.2e-6)
        ends = np.flatnonzero(on_sync[:-1] & ~on_sync[1:]) + 1
        vals = [np.median(raw[e + gate0:e + gate1]) for e in ends[:200] if e + gate1 < len(raw)]
        if vals:
            blank_level = float(np.median(vals))
    if found_porch:
        return float(sync_min), float(blank_level)
    return None, None

def scale_to_16bit_override(demod_input, outlinecount, outlinelen, agc_params):
    """Убирает построчный дрейф уровня гашения и масштабирует в 16 бит."""
    line_blank_levels = np.empty(outlinecount)
    sample_10us = max(1, int(8.0 * outlinelen / 64.0))
    sample_20us = max(sample_10us + 1, int(9.5 * outlinelen / 64.0))

    for i in range(outlinecount):
        start = i * outlinelen
        end = min(start + sample_20us, len(demod_input))
        if end > start + sample_10us:
            line_blank_levels[i] = np.median(demod_input[start + sample_10us : end])
        else:
            line_blank_levels[i] = agc_params.get("blank_level", 0.0)

    reduced = demod_input.astype(np.float64)

    # вычитаем дрейф
    for i in range(7, outlinecount - 5):
        start = i * outlinelen
        end = min((i + 1) * outlinelen, len(reduced))
        if end > start:
            avg_prev = np.median(line_blank_levels[max(0, i-2) : i+3])
            avg_next = np.median(line_blank_levels[max(0, i-1) : i+4])
            reduction = np.linspace(avg_prev, avg_next, num=(end - start))
            reduced[start:end] -= reduction[:end-start]

    # нормировка на IRE
    ire_unit = agc_params.get("ire_unit", 1.0)
    if ire_unit <= 0:
        ire_unit = 1.0
    reduced_ire = reduced / ire_unit

    # в 16 бит
    from constants import OUT_SCALE, OUTPUT_ZERO
    result = np.clip((reduced_ire * OUT_SCALE) + OUTPUT_ZERO, 0, 65535).astype(np.uint16)
    return result
