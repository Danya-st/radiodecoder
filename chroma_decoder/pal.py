"""PAL хрома-декодер: чётность поля (V-switch) по приращению фазы burst."""
import numpy as np
import scipy.signal as sps


def _wrap_pi(angle):
    """Угол в диапазон (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class PALChromaDecoder:
    def __init__(self, sample_rate):
        self.fs = sample_rate
        self.fsc = 4.43361875e6

        # окна от начала строки, с
        self.burst_start_s = 5.6e-6
        self.burst_end_s = 7.85e-6
        self.active_start_s = 10.4e-6

        # в отсчётах
        self.burst_start = int(self.burst_start_s * self.fs)
        self.burst_end = int(self.burst_end_s * self.fs)
        self.active_start = int(self.active_start_s * self.fs)

        nyq = self.fs / 2.0

        # полосовой фильтр цветности, fsc +-1 МГц
        bpf_low = (self.fsc - 1.0e6) / nyq
        bpf_high = (self.fsc + 1.0e6) / nyq
        self.bpf_b, self.bpf_a = sps.butter(3, [bpf_low, bpf_high], btype='band')

        # ФНЧ базовой полосы, 1.3 МГц
        self.lpf_b, self.lpf_a = sps.butter(4, 1.3e6 / nyq, btype='low')

    def measure_burst_phase(self, line):
        """Возвращает (фаза в радианах, амплитуда) burst строки; line начинается с синхроимпульса."""
        burst = line[self.burst_start:self.burst_end].astype(np.float64)
        if len(burst) < 8:
            return 0.0, 0.0
        burst = burst - np.mean(burst)
        window = np.hanning(len(burst))
        n = np.arange(self.burst_start, self.burst_end)
        osc = np.exp(-1j * 2.0 * np.pi * self.fsc * n / self.fs)
        z = 2.0 * np.sum(burst * window * osc) / np.sum(window)
        phase = np.angle(z) + np.pi / 2
        amplitude = np.abs(z)
        return _wrap_pi(phase), amplitude

    def demodulate_line(self, line, v_switch, comb_chroma=None):
        """
        Демодулирует строку, возвращает (Y, U, V) активной части.
        v_switch: знак V-компоненты (+1 или -1), comb_chroma: цветность от гребёнки (comb.py).
        """
        # фаза burst
        burst_phase, burst_amp = self.measure_burst_phase(line)

        if burst_amp < 0.01:
            Y = line[self.active_start:]
            return Y, np.zeros_like(Y), np.zeros_like(Y)

        # цветность
        chroma = sps.filtfilt(self.bpf_b, self.bpf_a, line if comb_chroma is None else comb_chroma)

        # несущие
        t = np.arange(len(line)) / self.fs

        adj_phase = burst_phase - v_switch*np.deg2rad(135.0)

        cos_ref = np.cos(2 * np.pi * self.fsc * t + adj_phase)
        sin_ref = np.sin(2 * np.pi * self.fsc * t + adj_phase)

        # синхронное детектирование
        corr_cos_raw = chroma * cos_ref * 2   # ~ V_true * v_switch
        corr_sin_raw = chroma * sin_ref * 2   # ~ U_true

        # ФНЧ
        corr_cos = sps.filtfilt(self.lpf_b, self.lpf_a, corr_cos_raw)
        U = sps.filtfilt(self.lpf_b, self.lpf_a, corr_sin_raw)

        # снимаем V-switch
        V = v_switch * corr_cos

        # яркость = композит минус цветность
        Y = line - chroma

        # активная часть
        Y_act = Y[self.active_start:]
        U_act = U[self.active_start:]
        V_act = V[self.active_start:]

        return Y_act, U_act, V_act

    def decode_field(self, field_data, linelocs, comb_chroma=None):
        """
        Декодирует поле с PAL delay line, возвращает (Y, U, V).
        field_data: float64, нормировка 100 IRE = 1; linelocs: начала строк;
        comb_chroma: цветность [n_lines, line_len] от гребёнки (comb.separate_field).
        """
        n_lines = len(linelocs) - 1
        lines = []
        burst_phase = np.zeros(n_lines)
        burst_amp = np.zeros(n_lines)

        # проход 1: фаза burst на каждой строке
        for i in range(n_lines):
            start = int(linelocs[i])
            end = int(linelocs[i + 1])
            line = field_data[start:end]
            lines.append(line)
            burst_phase[i], burst_amp[i] = self.measure_burst_phase(line)

            if i < 10 or i % 50 == 0:
                print(f"Строка {i}: burst_amp = {burst_amp[i]:.4f}, phase = {np.degrees(burst_phase[i]):.1f}")

        cycles_per_line = len(lines[0]) / 4.0 if lines else 0.0
        d_nom = 2.0 * np.pi * (cycles_per_line % 1.0)
        idx = np.arange(n_lines)
        good = burst_amp > max(1e-6, 0.3 * np.median(burst_amp[burst_amp > 0])) \
            if np.any(burst_amp > 0) else np.zeros(n_lines, dtype=bool)
        pair_ok = good[1:] & good[:-1]

        def hypothesis_score(even_sign):
            s = np.where(idx % 2 == 0, even_sign, -even_sign)
            eps = _wrap_pi(burst_phase - s * np.deg2rad(135.0))
            d = _wrap_pi(eps[1:] - eps[:-1])
            if not pair_ok.any():
                return -1.0
            return float(np.mean(np.cos(d[pair_ok] - d_nom)))

        score_plus = hypothesis_score(+1.0)
        score_minus = hypothesis_score(-1.0)
        even_sign = 1.0 if score_plus >= score_minus else -1.0
        v_switch_field = np.where(idx % 2 == 0, even_sign, -even_sign)


        decoded = [
            self.demodulate_line(lines[i], v_switch_field[i],
                                 None if comb_chroma is None else comb_chroma[i])
            for i in range(n_lines)
        ]

        # длина активной части - минимум по строкам
        active_len = min(len(d[0]) for d in decoded)
        Y_out = np.zeros((n_lines, active_len))
        U_out = np.zeros((n_lines, active_len))
        V_out = np.zeros((n_lines, active_len))

        # проход 2: PAL delay line
        for i in range(n_lines):
            Y_out[i] = decoded[i][0][:active_len]

            if i == 0:
                U_out[i] = decoded[i][1][:active_len]
                V_out[i] = decoded[i][2][:active_len]
            else:
                U_prev = decoded[i - 1][1][:active_len]
                V_prev = decoded[i - 1][2][:active_len]
                U_curr = decoded[i][1][:active_len]
                V_curr = decoded[i][2][:active_len]

                # усреднение с предыдущей строкой гасит фазовые ошибки
                U_out[i] = (U_curr + U_prev) / 2.0
                V_out[i] = (V_curr + V_prev) / 2.0

        return Y_out, U_out, V_out
