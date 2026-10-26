"""NTSC хрома-декодер: I/Q-модуляция, burst на Q-оси (180°)."""
import numpy as np
import scipy.signal as sps
from dataclasses import dataclass


@dataclass
class ChromaParams:
    """Параметры цветовой поднесущей."""
    fsc_hz: float              # частота цветовой поднесущей
    active_start_us: float     # начало активной области (мкс)
    active_end_us: float       # конец активной области (мкс)
    burst_start_us: float = 5.5
    burst_end_us: float = 7.8


class BurstDetector:
    """Фаза и амплитуда burst (~9 периодов fsc на заднем крыльце) - опора для синхронного детектора."""
    def __init__(self, chroma_params, sample_rate):
        self.params = chroma_params
        self.fs = sample_rate
        self.fsc = chroma_params.fsc_hz

        self.burst_start = int(chroma_params.burst_start_us * 1e-6 * sample_rate)
        self.burst_end = int(chroma_params.burst_end_us * 1e-6 * sample_rate)


    def detect_phase(self, line):
        """Возвращает (фаза, амплитуда) burst в строке."""
        burst = line[self.burst_start:self.burst_end]
        if len(burst) < 8:
            return 0.0, 0.0

        # корреляция с опорной синусоидой
        t = np.arange(self.burst_start, self.burst_end) / self.fs
        cos_ref = np.cos(2 * np.pi * self.fsc * t)
        sin_ref = np.sin(2 * np.pi * self.fsc * t)

        I = np.sum(burst * cos_ref) * 2 / len(burst)
        Q = np.sum(burst * sin_ref) * 2 / len(burst)

        phase = np.arctan2(I, Q)
        amplitude = np.sqrt(I**2 + Q**2)
        return phase, amplitude

    def make_carriers(self, line_length, phase):
        """Несущие (cos, sin), сфазированные по burst."""
        t = np.arange(line_length) / self.fs
        cos_carrier = np.cos(2 * np.pi * self.fsc * t + phase)
        sin_carrier = np.sin(2 * np.pi * self.fsc * t + phase)
        return cos_carrier, sin_carrier


NTSC_CHROMA = ChromaParams(
    fsc_hz=3.579545e6,
    active_start_us=10.0,
    active_end_us=62.0,
    burst_start_us=5.5,
    burst_end_us=7.8,
)

# уровень чёрного NTSC (7.5 IRE), см. constants.py
NTSC_SETUP = 0.075

# поворот I/Q -> U/V, 33°
NTSC_IQ_ANGLE = -33.0 * np.pi / 180.0


class NTSCDecoder:
    """Декодер строки NTSC."""
    def __init__(self, sample_rate):
        self.params = NTSC_CHROMA
        self.fs = sample_rate
        self.fsc = self.params.fsc_hz
        self.burst_det = BurstDetector(self.params, sample_rate)

        active_start = int(self.params.active_start_us * 1e-6 * sample_rate)
        active_end = int(self.params.active_end_us * 1e-6 * sample_rate)
        self.active_start = active_start
        self.active_end = active_end

        nyq = sample_rate / 2.0

        # ФНЧ базовой полосы: I до 1.5 МГц
        self.chroma_lpf_b, self.chroma_lpf_a = sps.butter(
            4, 1.5e6 / nyq, btype='low'
        )

        # полосовой фильтр цветности, fsc +-1 МГц
        low = (self.fsc - 1.0e6) / nyq
        high = (self.fsc + 1.0e6) / nyq
        self.chroma_bpf_b, self.chroma_bpf_a = sps.butter(
            4, [low, high], btype='band'
        )

        # Q уже по полосе, 0.5 МГц
        self.q_lpf_b, self.q_lpf_a = sps.butter(
            4, 0.5e6 / nyq, btype='low'
        )

    @staticmethod
    def _remove_setup(y):
        """Чёрный (7.5 IRE) -> 0, белый -> 1."""
        return np.clip((y - NTSC_SETUP) / (1.0 - NTSC_SETUP), 0.0, 1.0)

    def extract_chroma(self, line):
        """Цветность полосовым фильтром."""
        return sps.filtfilt(self.chroma_bpf_b, self.chroma_bpf_a, line)

    def lowpass_chroma(self, signal):
        """ФНЧ базовой полосы."""
        return sps.filtfilt(self.chroma_lpf_b, self.chroma_lpf_a, signal)

    def decode_line(self, line, comb_chroma=None):
        """
        Декодирует строку, возвращает (Y, U, V) активной части.
        comb_chroma: цветность строки от гребёнки (comb.py); берётся вместо композита
        и тоже идёт через полосовой фильтр.
        """
        # фаза burst
        phase, amp = self.burst_det.detect_phase(line)
        if amp < 0.01:
            Y = self._remove_setup(line)[self.active_start:self.active_end]
            # нет burst, строка ч/б
            return Y, np.zeros_like(Y), np.zeros_like(Y)

        # цветность
        chroma = self.extract_chroma(line if comb_chroma is None else comb_chroma)
        carrier_phase = phase + np.pi - NTSC_IQ_ANGLE

        # несущие по burst (он на Q-оси, 180°)
        cos_carrier, sin_carrier = self.burst_det.make_carriers(
            len(line), carrier_phase
        )

        # синхронное детектирование
        I_raw = chroma * cos_carrier * 2
        Q_raw = chroma * sin_carrier * 2

        # ФНЧ
        I = self.lowpass_chroma(I_raw)
        Q = sps.filtfilt(self.q_lpf_b, self.q_lpf_a, Q_raw)

        # I/Q -> U/V
        cos_a = np.cos(NTSC_IQ_ANGLE)
        sin_a = np.sin(NTSC_IQ_ANGLE)
        U = sin_a * I + cos_a * Q
        V = cos_a * I - sin_a * Q

        # яркость = композит минус цветность
        Y = self._remove_setup(line - chroma)

        return (
            Y[self.active_start:self.active_end],
            U[self.active_start:self.active_end],
            V[self.active_start:self.active_end],
        )
