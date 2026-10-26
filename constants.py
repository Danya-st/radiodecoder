"""Параметры PAL и NTSC (ITU-R BT.470, SMPTE 170M) и color-under для VHS."""

# PAL (625 строк, 50 Гц)
PAL_SYS = {
    "name": "PAL",
    "fsc_mhz": 4.43361875,
    "line_period": 64.0,  # мкс
    "frame_lines": 625,
    "field_lines": (312, 313),
    "active_lines": 288,
    "vsync_ire": -40,
    "sync_ire": -40.0,
    "setup_ire": 0.0,
    "burst_ire": 43.0,      # размах (p-p), 300 мВ из 700 мВ
    "color_burst_us": (5.6, 7.85),
    "active_video_us": (10.5, 62.5),
    "hsync_pulse_us": 4.7,
    "num_pulses": 5,
    "chroma": "pal",
    "field_lines_tbc": 313
}

# NTSC (525 строк, 60 Гц)
NTSC_SYS = {
    "name": "NTSC",
    "fsc_mhz": 315.0 / 88.0,  # ~3.579545 МГц
    "line_period": 63.555555,  # мкс (227.5 цветовых циклов)
    "frame_lines": 525,
    "field_lines": (263, 262),
    "active_lines": 240,
    "vsync_ire": -40.0,
    "sync_ire": -40.0,
    "setup_ire": 7.5,
    "burst_ire": 40.0,      # размах (p-p)
    "color_burst_us": (5.3, 7.8),
    "active_video_us": (9.45, 62.55),
    "hsync_pulse_us": 4.7,
    "num_pulses": 6,
    "chroma": "ntsc",
    "field_lines_tbc": 263
}

OUTPUT_ZERO = 16384.0      # код, соответствующий 0 IRE
OUT_SCALE = 327.68         # кодов на 1 IRE


def get_system(system_name: str) -> dict:
    """Параметры системы по имени ("PAL" или "NTSC", регистр не важен)."""
    system_name = system_name.upper().strip()

    if system_name == "PAL":
        return PAL_SYS
    elif system_name == "NTSC":
        return NTSC_SYS
    else:
        raise ValueError(f"Неизвестная видеосистема: '{system_name}'. "
                         f"Поддерживаемые: PAL, NTSC")


FM_HZ_PER_VOLT = 1.0e6

VHS_UNDERCOLOR = {
    "pal": {
        "fsc": 4.43361875e6,
        "under_freq": 40.125 * 15625.0,
        "fm_sync": 3.8e6,
        "sync_frac": 0.3,
        "volts_per_unit": 0.7,        # 100 IRE = 0.7 В
    },
    "ntsc": {
        "fsc": 315.0e6 / 88.0,
        "under_freq": 40.0 * (315.0e6 / 88.0) * 2.0 / 455.0,
        "fm_sync": 3.4e6,
        "sync_frac": 40.0 / 140.0,
        "volts_per_unit": 1.0 / 1.4,  # 100 IRE = 1/1.4 В
    },
}
