"""Предобработка CVBS. Демодуляции нет, вход уже базовая полоса."""
import scipy.signal as sps
from filters import generate_f05_filter
from scipy.ndimage import median_filter

class RFDecode:
    def __init__(self, sys_params, sample_rate_mhz):
        self.sample_rate = sample_rate_mhz * 1e6

        self.f05_b, self.f05_a = generate_f05_filter(self.sample_rate)

        self.agc_params = {
            "blank_level": 0.0,
            "ire_unit": 1.0
        }

    def demodblock(self, data):
        """Исходный сигнал и он же после ФНЧ 0.5 МГц."""
        # исходный сигнал
        luma = data

        # ФНЧ 0.5 МГц после медианного фильтра, по нему ищем синхро
        k = max(3, int(round(self.sample_rate * 0.4e-6)) | 1)
        luma05 = sps.filtfilt(self.f05_b, self.f05_a, median_filter(luma, size=k, mode="nearest"))

        return {
            "video": {
                "demod": luma,
                "demod_05": luma05
            }
        }
