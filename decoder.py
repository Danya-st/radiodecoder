"""Декодер CVBS: фрагмент сигнала -> поле."""
import numpy as np
from rfdecode import RFDecode
from field import FieldCVBS
from constants import PAL_SYS, NTSC_SYS

class CVBSDecoder:
    def __init__(self, system="PAL", sample_rate_mhz=None):
        self.system = system.upper()
        if self.system == "PAL":
            self.sys_params = PAL_SYS
            self.sample_rate_mhz = sample_rate_mhz or (4 * self.sys_params["fsc_mhz"])
        elif self.system == "NTSC":
            self.sys_params = NTSC_SYS
            self.sample_rate_mhz = sample_rate_mhz or (4 * self.sys_params["fsc_mhz"])
        else:
            raise ValueError(f"Неизвестная система: {self.system}")

        self.rf = RFDecode(self.sys_params, self.sample_rate_mhz)
        self.inlinelen = self.rf.sample_rate * (self.sys_params["line_period"] / 1e6)

    def decode_field(self, data_chunk):
        """Декодирует одно поле из фрагмента CVBS."""
        data = data_chunk.astype(np.float64)

        demod_data = self.rf.demodblock(data)

        field = FieldCVBS(self.rf, demod_data, self.sys_params)
        if not field.process():
            return None

        return field
