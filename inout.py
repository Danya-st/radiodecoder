"""Чтение .wav и сырых файлов (.r16, .s16), запись .tbc."""
import numpy as np
from scipy.io import wavfile

def read_input(filename):
    """Возвращает (данные float64, частота дискретизации в Гц)."""
    if filename.lower().endswith('.wav'):
        sample_rate_hz, data = wavfile.read(filename)

        # из многоканального берём первый канал
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = data[:, 0]

        return data.astype(np.float64), sample_rate_hz

    else:
        # сырой файл (.cvbs, .r16, .s16), читаем как uint8
        data = np.fromfile(filename, dtype=np.uint8)

        # по умолчанию PAL 4fsc
        sample_rate = 17.734475e6

        return data.astype(np.float64), sample_rate


def write_tbc(filename, field_data):
    """Дописывает поле в конец .tbc."""
    with open(filename, 'ab') as f:
        f.write(field_data.tobytes())
