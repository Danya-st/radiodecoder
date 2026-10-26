#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: fm_demod
# GNU Radio version: 3.10.12.0
#
# Два режима (флаг undercolor):
#   False - цветность на месте внутри композита (LaserDisc и т.п.). Вход - I/Q,
#           ЧМ-квадратурный демодулятор -> ФНЧ -> CVBS.
#   True  - цветность записана под ЧМ-яркостью (VHS). Вход - вещественный ВЧ-сигнал в канале 0
#           (канал 1 не используется, но файл должен быть 2-канальным).
#           Яркость: ПФ ЧМ-полосы -> Гильберт -> ЧМ-демодулятор -> вычитание частоты чёрного.
#           Цветность: ФНЧ ~1.2 МГц -> гетеродин fsc+f_under -> ПФ около fsc -> chroma_gain.
#           Сумма даёт обычный CVBS. Построчный поворот фазы цветности не снимается, см. README.
# Выход - float WAV (CVBS) для main.py.

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float
from gnuradio import eng_notation
import os
import threading

from constants import VHS_UNDERCOLOR, FM_HZ_PER_VOLT   



class fm_demod(gr.top_block, Qt.QWidget):

    def __init__(self, chroma_gain=10.0, deviation=5e6, input_file='samples/test_5s.wav', output_file='output2.wav', repeat=True, samp_rate=20e6, system='pal', undercolor=False, video_bw=6e6):
        gr.top_block.__init__(self, "fm_demod", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("fm_demod")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "fm_demod")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        # из командной строки приходит float, а блокам нужен int
        samp_rate = int(samp_rate)

        # параметры
        self.chroma_gain = chroma_gain
        self.deviation = deviation
        self.input_file = input_file
        self.output_file = output_file
        self.repeat = repeat
        self.samp_rate = samp_rate
        self.system = system
        self.undercolor = undercolor
        self.video_bw = video_bw

        # переменные
        self.vhs = vhs = VHS_UNDERCOLOR[system.lower()]
        self.quad_gain = quad_gain = samp_rate / (2 * 3.1415926535 * deviation)
        self.uc_quad_gain = uc_quad_gain = samp_rate / (2 * 3.1415926535 * FM_HZ_PER_VOLT)
        self.fm_black = fm_black = vhs["fm_sync"] + vhs["sync_frac"] * FM_HZ_PER_VOLT
        self.lo_freq = lo_freq = vhs["fsc"] + vhs["under_freq"]
        self.luma_rf_low = luma_rf_low = 2.0e6
        self.luma_rf_high = luma_rf_high = min(8.0e6, samp_rate / 2 - 0.5e6)
        self.luma_rf_taps = luma_rf_taps = firdes.band_pass(1, samp_rate, luma_rf_low, luma_rf_high, 0.5e6, window.WIN_HAMMING, 6.76)
        self.hilbert_ntaps = hilbert_ntaps = 65
        self.chroma_lpf_taps = chroma_lpf_taps = firdes.low_pass(1, samp_rate, 1.2e6, 0.4e6, window.WIN_HAMMING, 6.76)
        self.chroma_bpf_taps = chroma_bpf_taps = firdes.band_pass(1, samp_rate, vhs["fsc"] - 0.8e6, vhs["fsc"] + 0.6e6, 0.25e6, window.WIN_HAMMING, 6.76)
        # групповые задержки КИХ, нужны для выравнивания веток
        self.luma_delay = luma_delay = (len(luma_rf_taps) - 1) // 2 + (hilbert_ntaps - 1) // 2 + 1
        # переходный процесс КИХ даёт выбросы и ломает поиск синхро в main.py, пропускаем
        self.startup_skip = startup_skip = 4096
        self.chroma_delay = chroma_delay = (len(chroma_lpf_taps) - 1) // 2 + (len(chroma_bpf_taps) - 1) // 2

        # блоки

        # чекбокс undercolor
        _undercolor_check_box = Qt.QCheckBox("'undercolor' (цветность под яркостью)")
        self._undercolor_choices = {True: True, False: False}
        self._undercolor_choices_inv = dict((v, k) for k, v in self._undercolor_choices.items())
        self._undercolor_callback = lambda i: Qt.QMetaObject.invokeMethod(_undercolor_check_box, "setChecked", Qt.Q_ARG("bool", self._undercolor_choices_inv[i]))
        self._undercolor_callback(self.undercolor)
        _undercolor_check_box.stateChanged.connect(lambda i: self.set_undercolor(self._undercolor_choices[bool(i)]))
        self.top_layout.addWidget(_undercolor_check_box)
        # слайдер chroma_gain
        self._chroma_gain_range = qtgui.Range(0, 50, 0.1, 10.0, 200)
        self._chroma_gain_win = qtgui.RangeWidget(self._chroma_gain_range, self.set_chroma_gain, "'chroma_gain'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._chroma_gain_win)

        self.low_pass_filter_0 = filter.fir_filter_fff(
            1,
            firdes.low_pass(
                1,
                samp_rate,
                video_bw,
                (video_bw/10),
                window.WIN_HAMMING,
                6.76))
        self.blocks_wavfile_source_0 = blocks.wavfile_source(input_file, repeat)
        self.blocks_wavfile_sink_0 = blocks.wavfile_sink(
            output_file,
            1,
            samp_rate,
            blocks.FORMAT_WAV,
            blocks.FORMAT_FLOAT,
            False
            )
        self.blocks_skiphead_0 = blocks.skiphead(gr.sizeof_float*1, startup_skip)
        self.blocks_selector_0 = blocks.selector(gr.sizeof_float*1, int(bool(undercolor)), 0)
        self.blocks_selector_0.set_enabled(True)

        # ветка 0 (undercolor=False): обычная ЧМ-демодуляция
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.analog_quadrature_demod_cf_0 = analog.quadrature_demod_cf(quad_gain)

        # ветка 1 (undercolor=True): яркость
        self.uc_luma_bpf = filter.fir_filter_fff(1, luma_rf_taps)
        self.uc_hilbert = filter.hilbert_fc(hilbert_ntaps, window.WIN_HAMMING, 6.76)
        self.uc_fm_demod = analog.quadrature_demod_cf(uc_quad_gain)
        self.uc_black_offset = blocks.add_const_ff((-fm_black / FM_HZ_PER_VOLT))
        self.uc_luma_delay = blocks.delay(gr.sizeof_float*1, max(chroma_delay - luma_delay, 0))

        # ветка 1: цветность
        self.uc_chroma_lpf = filter.fir_filter_fff(1, chroma_lpf_taps)
        self.uc_lo = analog.sig_source_f(samp_rate, analog.GR_COS_WAVE, lo_freq, 2.0, 0, 0)
        self.uc_mixer = blocks.multiply_ff(1)
        self.uc_chroma_bpf = filter.fir_filter_fff(1, chroma_bpf_taps)
        self.uc_chroma_gain = blocks.multiply_const_ff(chroma_gain)
        self.uc_chroma_delay = blocks.delay(gr.sizeof_float*1, max(luma_delay - chroma_delay, 0))
        self.uc_sum = blocks.add_ff(1)


        # соединения
        # вход: два канала WAV
        self.connect((self.blocks_wavfile_source_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.blocks_wavfile_source_0, 1), (self.blocks_float_to_complex_0, 1))
        # ветка 0
        self.connect((self.blocks_float_to_complex_0, 0), (self.analog_quadrature_demod_cf_0, 0))
        self.connect((self.analog_quadrature_demod_cf_0, 0), (self.blocks_selector_0, 0))
        # ветка 1: яркость (ВЧ из канала 0)
        self.connect((self.blocks_wavfile_source_0, 0), (self.uc_luma_bpf, 0))
        self.connect((self.uc_luma_bpf, 0), (self.uc_hilbert, 0))
        self.connect((self.uc_hilbert, 0), (self.uc_fm_demod, 0))
        self.connect((self.uc_fm_demod, 0), (self.uc_black_offset, 0))
        self.connect((self.uc_black_offset, 0), (self.uc_luma_delay, 0))
        self.connect((self.uc_luma_delay, 0), (self.uc_sum, 0))
        # ветка 1: цветность (ВЧ из канала 0)
        self.connect((self.blocks_wavfile_source_0, 0), (self.uc_chroma_lpf, 0))
        self.connect((self.uc_chroma_lpf, 0), (self.uc_mixer, 0))
        self.connect((self.uc_lo, 0), (self.uc_mixer, 1))
        self.connect((self.uc_mixer, 0), (self.uc_chroma_bpf, 0))
        self.connect((self.uc_chroma_bpf, 0), (self.uc_chroma_gain, 0))
        self.connect((self.uc_chroma_gain, 0), (self.uc_chroma_delay, 0))
        self.connect((self.uc_chroma_delay, 0), (self.uc_sum, 1))
        self.connect((self.uc_sum, 0), (self.blocks_selector_0, 1))
        # выход
        self.connect((self.blocks_selector_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.blocks_skiphead_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.blocks_wavfile_sink_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "fm_demod")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_chroma_gain(self):
        return self.chroma_gain

    def set_chroma_gain(self, chroma_gain):
        self.chroma_gain = chroma_gain
        self.uc_chroma_gain.set_k(self.chroma_gain)

    def get_deviation(self):
        return self.deviation

    def set_deviation(self, deviation):
        self.deviation = deviation
        self.set_quad_gain(self.samp_rate / (2 * 3.1415926535 * self.deviation))

    def get_input_file(self):
        return self.input_file

    def set_input_file(self, input_file):
        self.input_file = input_file

    def get_output_file(self):
        return self.output_file

    def set_output_file(self, output_file):
        self.output_file = output_file

    def get_repeat(self):
        return self.repeat

    def set_repeat(self, repeat):
        self.repeat = repeat

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        # на лету частоту менять нельзя, она задаётся при запуске (--samp-rate)
        self.samp_rate = samp_rate

    def get_system(self):
        return self.system

    def get_undercolor(self):
        return self.undercolor

    def set_undercolor(self, undercolor):
        self.undercolor = undercolor
        self._undercolor_callback(self.undercolor)
        self.blocks_selector_0.set_input_index(int(bool(self.undercolor)))

    def get_video_bw(self):
        return self.video_bw

    def set_video_bw(self, video_bw):
        self.video_bw = video_bw
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, self.video_bw, (self.video_bw/10), window.WIN_HAMMING, 6.76))

    def get_quad_gain(self):
        return self.quad_gain

    def set_quad_gain(self, quad_gain):
        self.quad_gain = quad_gain
        self.analog_quadrature_demod_cf_0.set_gain(self.quad_gain)



def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", dest="input_file", type=str, default='samples/test_5s.wav',
        help="Входной 2-канальный WAV (I/Q; для --undercolor - ВЧ в канале 0) [default=%(default)r]")
    parser.add_argument(
        "-o", "--output", dest="output_file", type=str, default='output2.wav',
        help="Выходной WAV (float, CVBS) [default=%(default)r]")
    parser.add_argument(
        "--samp-rate", dest="samp_rate", type=eng_float, default=eng_notation.num_to_str(float(20e6)),
        help="Частота дискретизации входного WAV, Гц [default=%(default)r]")
    parser.add_argument(
        "--deviation", dest="deviation", type=eng_float, default=eng_notation.num_to_str(float(5e6)),
        help="Девиация ЧМ (только без --undercolor), Гц [default=%(default)r]")
    parser.add_argument(
        "--video-bw", dest="video_bw", type=eng_float, default=eng_notation.num_to_str(float(6e6)),
        help="Полоса ФНЧ после демодулятора, Гц [default=%(default)r]")
    parser.add_argument(
        "--undercolor", dest="undercolor", action="store_true", default=False,
        help="Цветность записана под ЧМ-яркостью (color-under): вернуть её на поднесущую")
    parser.add_argument(
        "--system", dest="system", type=str, default='pal', choices=['pal', 'ntsc'],
        help="Система (задаёт частоты color-under, только с --undercolor) [default=%(default)r]")
    parser.add_argument(
        "--chroma-gain", dest="chroma_gain", type=eng_float, default=eng_notation.num_to_str(float(10.0)),
        help="Усиление восстановленной цветности (только с --undercolor) [default=%(default)r]")
    parser.add_argument(
        "--headless", dest="headless", action="store_true", default=False,
        help="Без окна: обработать файл целиком (без зацикливания) и выйти")
    return parser


def main(top_block_cls=fm_demod, options=None):
    if options is None:
        options = argument_parser().parse_args()

    if options.headless:
        # Qt-виджетам нужен QApplication, но не нужен дисплей
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(chroma_gain=options.chroma_gain, deviation=options.deviation,
                       input_file=options.input_file, output_file=options.output_file,
                       repeat=not options.headless, samp_rate=options.samp_rate,
                       system=options.system, undercolor=options.undercolor,
                       video_bw=options.video_bw)

    if options.headless:
        tb.start()
        tb.flowgraph_started.set()
        tb.wait()
        return

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
