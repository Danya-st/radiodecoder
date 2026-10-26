"""Фильтры для поиска синхро и выделения burst."""
import scipy.signal as sps

def generate_f05_filter(sample_rate_hz):
    """ФНЧ 0.5 МГц для поиска синхроимпульсов. Возвращает (b, a) для filtfilt."""
    nyq = sample_rate_hz / 2.0
    b, a = sps.butter(4, 0.5e6 / nyq, btype='low')
    return b, a
