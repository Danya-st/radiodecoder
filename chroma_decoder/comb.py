"""
Гребенчатый Y/C-фильтр для CVBS в TBC-сетке 4*fsc (NTSC и PAL).

Если фаза поднесущей в двух строках отличается на 180°, цветность в них противофазна,
а яркость почти одинакова: luma = (a + b) / 2, chroma = (a - b) / 2.

Система определяется по сдвигу фазы между строками, 360° * frac(line_len / 4):
    NTSC: 910 отсчётов, сдвиг 180°, парная строка соседняя (±1)
    PAL: 1135 отсчётов, сдвиг 270° и V инвертируется (V-switch), поэтому берём строку
         через одну (±2): 2*270° = 180°, знак V тот же. Вертикальное разрешение ниже.

Внутри поля фильтр трёхотводный, chroma = (2*cur - prev - next) / 4; у краёв двухстрочный.
Если соседей нет вовсе, цветность не выделяется.
"""
import numpy as np


class CombFilter2D:
    """
    Гребенчатый фильтр.
        separate_field(field)      2D-массив (строки x отсчёты) -> (luma, chroma)
        separate(lines, linelocs)  плоский массив и позиции строк -> списки строк
    """

    def __init__(self, sample_rate, fsc_hz):
        self.fs = sample_rate
        self.fsc = fsc_hz
        self.system = None   # "NTSC" / "PAL", по первой строке
        self.step = None     # шаг до парной строки (1 или 2)

    @staticmethod
    def detect_system(line_len):
        """Возвращает ("NTSC", 1) или ("PAL", 2): система и шаг до строки с противофазной цветностью."""
        shift_deg = 360.0 * ((line_len / 4.0) % 1.0)
        if abs(shift_deg - 180.0) <= 1.0:
            return "NTSC", 1
        if abs(shift_deg - 270.0) <= 1.0 or abs(shift_deg - 90.0) <= 1.0:
            return "PAL", 2
        raise NotImplementedError(
            f"CombFilter2D: межстрочный сдвиг поднесущей ~{shift_deg:.1f}° "
            f"(line_len={line_len}) не соответствует ни NTSC (180°), ни PAL (270°)."
        )

    def separate_field(self, field):
        """field: массив [n_lines, line_len], одна TBC-строка на строку. Возвращает (luma, chroma) той же формы."""
        field = np.asarray(field, dtype=np.float64)
        n_lines, line_len = field.shape
        if self.system is None:
            self.system, self.step = self.detect_system(line_len)
        k = self.step

        chroma = np.zeros_like(field)
        if n_lines <= k:
            return field.copy(), chroma

        idx = np.arange(n_lines)
        has_prev = idx - k >= 0
        has_next = idx + k < n_lines
        prev = field[np.clip(idx - k, 0, n_lines - 1)]
        nxt = field[np.clip(idx + k, 0, n_lines - 1)]

        both = has_prev & has_next
        only_prev = has_prev & ~has_next
        only_next = ~has_prev & has_next

        chroma[both] = (2.0 * field[both] - prev[both] - nxt[both]) / 4.0
        chroma[only_prev] = (field[only_prev] - prev[only_prev]) / 2.0
        chroma[only_next] = (field[only_next] - nxt[only_next]) / 2.0

        return field - chroma, chroma

    def separate(self, lines, linelocs):
        """lines: плоский массив; linelocs: начала строк (n_lines + 1). Возвращает (luma, chroma) списками строк."""
        n_lines = len(linelocs) - 1
        line_len = int(linelocs[1]) - int(linelocs[0])
        rows = np.stack([
            np.asarray(lines[int(linelocs[i]):int(linelocs[i]) + line_len], dtype=np.float64)
            for i in range(n_lines)
        ])
        luma, chroma = self.separate_field(rows)
        return list(luma), list(chroma)
