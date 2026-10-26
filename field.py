"""Поле CVBS: синхронизация и ресемплинг в TBC-сетку."""
import numpy as np
from sync import findpulses, compute_line_len, get_line0, valid_pulses_to_linelocs
from agc import find_sync_levels, scale_to_16bit_override

def _resample_field_lines(samples, linelocs, outlinelen, outlinecount):
    """Каждая строка [linelocs[i], linelocs[i+1]) линейной интерполяцией приводится к outlinelen отсчётам."""
    n = len(samples)
    out = np.empty((outlinecount, outlinelen), dtype=np.float64)
    frac = np.arange(outlinelen, dtype=np.float64) / outlinelen
    for i in range(outlinecount):
        start, end = linelocs[i], linelocs[i + 1]
        pos = np.clip(start + (end - start) * frac, 0.0, n - 1.0 - 1e-6)
        i0 = pos.astype(np.int64)
        f = pos - i0
        out[i] = samples[i0] * (1.0 - f) + samples[i0 + 1] * f
    return out


class FieldCVBS:
    def __init__(self, rf, data, sys_params):
        self.rf = rf
        self.data = data
        self.sys_params = sys_params
        fsc_hz = self.sys_params["fsc_mhz"] * 1e6
        self.outlinelen = int(np.round(4.0 * fsc_hz * (self.sys_params["line_period"] / 1e6)))

        self.outlinecount = sys_params["field_lines_tbc"]
        self.linelocs = None
        self.valid = False
        self.resampled = None

    def process(self):
        # уровни синхро и гашения
        sync_min, blank_level = find_sync_levels(
            self.data["video"]["demod_05"], self.rf.sample_rate, self.data["video"]["demod"])

        if sync_min is None or blank_level is None:
            return False

        self.rf.agc_params["blank_level"] = blank_level

        ire_unit = (blank_level - sync_min) / (-self.sys_params["vsync_ire"])
        self.rf.agc_params["ire_unit"] = ire_unit

        # пороги импульсов
        pulse_max = blank_level + (self.sys_params["vsync_ire"] / 2.0) * ire_unit

        # синхроимпульсы
        min_len = int(self.rf.sample_rate * 1.0e-6)    # короче 1 мкс - помеха
        max_len = int(self.rf.sample_rate * 35.0e-6)
        pulses = findpulses(self.data["video"]["demod_05"], pulse_max, min_len, max_len)

        # позиции строк
        meanlinelen = compute_line_len(pulses)

        line0loc = get_line0(pulses, meanlinelen, self.sys_params)

        if meanlinelen is None or line0loc is None:
            return False

        # запас строк, чтобы у последней строки был конец
        proclines = self.outlinecount + 10
        self.linelocs = valid_pulses_to_linelocs(pulses, line0loc, meanlinelen, proclines)

        # выравниваем строки (TBC)
        self.resampled = _resample_field_lines(
            self.data["video"]["demod"], self.linelocs, self.outlinelen, self.outlinecount
        )
        self.valid = True
        return True

    def scale_to_16bit(self, demod_input=None):
        """Масштабирует поле в 16 бит (формат TBC)."""
        if demod_input is None:
            if self.resampled is None:
                raise RuntimeError("Поле не обработано: вызовите process() перед scale_to_16bit()")
            demod_input = self.resampled.reshape(-1)
        return scale_to_16bit_override(
            demod_input,
            self.outlinecount,
            self.outlinelen,
            self.rf.agc_params
        )
