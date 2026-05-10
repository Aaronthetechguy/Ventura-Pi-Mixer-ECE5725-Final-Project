"""Pygame touchscreen GUI for the live multi-channel EQ mixer.

This file draws one expanded EQ editor, compact previews for every channel,
mute buttons, pan indicators, and master/track meters. It also translates
mouse and keyboard input into EQ state changes that rebuild the audio engine.
"""

import math
import time

import numpy as np
import pygame
from scipy.signal import sosfreqz

from biquad_eq3 import (
    butterworth_section_qs as biquad_butterworth_section_qs,
    make_high_shelf,
    make_highpass,
    make_low_shelf,
    make_lowpass,
    make_peaking_eq,
)
from eq_state import EQState


class EQChannelTile:
    """Reusable single-channel EQ tile.

    The parent layout class can create one instance per channel and call:
      - set_bounds(rect, expanded)
      - set_meter(meter_dict)
      - handle_event(event)
      - update(dt)
      - draw()

    handle_event returns an action dict:
      {"select_track": track_index} when a minimized tile is clicked.
      {} when no parent-level action is requested.
    """

    def __init__(self, screen, eq_state, audio_engine, track_index, label=None):
        # Each tile owns drawing and input for exactly one channel.
        self.screen = screen
        self.eq_state = eq_state
        self.audio_engine = audio_engine
        self.track_index = int(track_index)
        self.label = label or f"CH {self.track_index + 1}"
        self.disabled = False

        self.rect = pygame.Rect(0, 0, 0, 0)
        self.expanded = False

        self.header_rect = pygame.Rect(0, 0, 0, 0)
        self.graph_rect = pygame.Rect(0, 0, 0, 0)
        self.preview_rect = pygame.Rect(0, 0, 0, 0)
        self.q_slider_rect = pygame.Rect(0, 0, 0, 0)
        self.vu_rect = pygame.Rect(0, 0, 0, 0)
        self.mini_clip_rect = pygame.Rect(0, 0, 0, 0)        
        self.reset_button_rect = pygame.Rect(0, 0, 0, 0)
        self.reset_q_button_rect = pygame.Rect(0, 0, 0, 0)
        self.reset_default_button_rect = pygame.Rect(0, 0, 0, 0)
        self.type_button_rects = []
        self.order_button_rects = []
        self.meter_horizontal = False

        # Stronger VENUE / S6L-inspired palette.
        self.bg_color = (3, 8, 14)
        self.bg_active = (6, 18, 28)
        self.border_color = (44, 110, 150)
        self.border_active = (96, 224, 255)
        self.grid_color = (20, 58, 84)
        self.axis_color = (98, 182, 220)
        self.curve_color = (88, 242, 255)
        self.preview_curve_color = (64, 212, 246)
        self.text_color = (232, 242, 248)
        self.muted_text = (120, 154, 176)
        self.pan_left_color = (78, 154, 255)
        self.pan_center_color = (112, 214, 170)
        self.pan_right_color = (255, 126, 92)
        self.pan_track_color = (22, 36, 46)
        self.pan_center_glow = (178, 235, 205)

        self.band_colors = [
            (255, 112, 102),   # warm red/orange
            (115, 228, 120),   # green
            (132, 166, 255),   # blue
            (248, 194, 92),    # amber
        ]

        self.font = pygame.font.SysFont("DejaVu Sans", 18)
        self.small_font = pygame.font.SysFont("DejaVu Sans", 13)
        self.tiny_font = pygame.font.SysFont("DejaVu Sans", 11)

        self.selected_band = None
        self.dragging_band = False
        self.dragging_q = False

        self.curve_options = ["bell", "hp", "lp", "low_shelf", "high_shelf"]
        self.order_options = [2, 4, 6, 8]

        # Curve rendering is cached until a band or layout changes.
        self.q_slider_knob_radius = 9
        self.last_rebuild_time = 0.0
        self.preview_dim_alpha = 0
        self.preview_tint_alpha = 0

        self.min_freq = float(eq_state.min_freq)
        self.max_freq = float(eq_state.max_freq)
        self.min_gain = float(eq_state.min_gain)
        self.max_gain = float(eq_state.max_gain)

        self.log_min_freq = math.log10(self.min_freq)
        self.log_max_freq = math.log10(self.max_freq)
        self.log_freq_span = self.log_max_freq - self.log_min_freq

        self.curve_freqs = np.logspace(self.log_min_freq, self.log_max_freq, 420)
        self.curve_dirty = True
        self.cached_curve_points = []
        self._frame_bands = None

        self.meter = {
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "peak_hold_dbfs": -120.0,
            "clipped": False,
        }

    def set_preview_style(self, dimmed=False, muted_tint=False):
        """Set visual overlays used by minimized channel previews."""
        self.preview_dim_alpha = 88 if dimmed else 0
        self.preview_tint_alpha = 58 if muted_tint else 0

    def set_disabled(self, disabled):
        """Mark a tile disconnected when the audio engine has fewer tracks."""
        self.disabled = bool(disabled)

    def set_bounds(self, rect, expanded=False):
        """Place the tile and rebuild child rectangles."""
        self.rect = pygame.Rect(rect)
        self.expanded = bool(expanded)
        self._build_layout()
        self.curve_dirty = True

    def set_expanded(self, expanded):
        """Switch between full editor and compact preview mode."""
        expanded = bool(expanded)
        if expanded == self.expanded:
            return
        self.expanded = expanded
        self._build_layout()
        self.curve_dirty = True

    def set_meter(self, meter):
        """Update cached meter values from the audio engine."""
        if not meter:
            return
        self.meter = {
            "rms_dbfs": float(meter.get("rms_dbfs", -120.0)),
            "peak_dbfs": float(meter.get("peak_dbfs", -120.0)),
            "peak_hold_dbfs": float(meter.get("peak_hold_dbfs", meter.get("peak_dbfs", -120.0))),
            "clipped": bool(meter.get("clipped", False)),
        }

    def _build_layout(self):
        """Compute all child rectangles from the current tile size."""
        pad = 8

        if self.expanded:
            # Expanded mode reserves the lower area for controls.
            header_h = 28
        else:
            header_h = 14

        self.header_rect = pygame.Rect(
            self.rect.left + pad,
            self.rect.top + pad,
            self.rect.width - 2 * pad,
            header_h,
        )

        if self.expanded:
            meter_w = 12
            meter_gap = 22
            self.meter_horizontal = False

            # Reserve a dedicated bottom control zone first.
            footer_h = 108

            graph_left_pad = 34

            self.vu_rect = pygame.Rect(
                self.rect.right - pad - meter_w - 8,
                self.header_rect.bottom + 12,
                meter_w,
                self.rect.height - header_h - footer_h - 28,
            )

            self.graph_rect = pygame.Rect(
                self.rect.left + graph_left_pad,
                self.header_rect.bottom + 12,
                self.vu_rect.left - (self.rect.left + graph_left_pad) - meter_gap,
                self.rect.height - header_h - footer_h - 28,
            )

            # Q slider row.
            q_row_y = self.graph_rect.bottom + 30
            self.q_slider_rect = pygame.Rect(
                self.graph_rect.left + 40,
                q_row_y,
                self.graph_rect.width - 48,
                6,
            )

            # Type buttons row.
            type_y = q_row_y + 20
            type_w = 56
            type_h = 24
            type_gap = 8

            self.type_button_rects = []
            type_labels = ["bell", "hp", "lp", "LS", "HS"]

            x = self.graph_rect.left
            for label in type_labels:
                r = pygame.Rect(x, type_y, type_w, type_h)
                self.type_button_rects.append((label, r))
                x += type_w + type_gap

            # Order buttons row.
            order_y = type_y + 40
            order_w = 34
            order_h = 22
            order_gap = 8

            self.order_button_rects = []
            ox = self.graph_rect.left
            for order in self.order_options:
                r = pygame.Rect(ox, order_y, order_w, order_h)
                self.order_button_rects.append((order, r))
                ox += order_w + order_gap

            # Reset buttons on the right, aligned to rows.
            self.reset_button_rect = pygame.Rect(
                self.graph_rect.right - 120,
                order_y - 2,
                56,
                24,
            )
            self.reset_q_button_rect = pygame.Rect(
                self.graph_rect.right - 58,
                order_y - 2,
                50,
                24,
            )
            self.reset_default_button_rect = pygame.Rect(
                self.graph_rect.right - 176,
                order_y - 2,
                50,
                24,
            )

            self.preview_rect = pygame.Rect(0, 0, 0, 0)
            self.mini_clip_rect = pygame.Rect(0, 0, 0, 0)


        else:
            # Preview mode prioritizes a tiny curve, pan bar, and meter.
            self.meter_horizontal = True
            meter_h = 10
            clip_w = 7
            clip_gap = 3

            meter_w = self.rect.width - 2 * pad - clip_gap - clip_w
            meter_w = max(18, meter_w)

            self.vu_rect = pygame.Rect(
                self.rect.left + pad,
                self.rect.bottom - pad - meter_h,
                meter_w,
                meter_h,
            )

            self.mini_clip_rect = pygame.Rect(
                self.vu_rect.right + clip_gap,
                self.vu_rect.top + 1,
                clip_w,
                self.vu_rect.height - 2,
            )

            preview_y = self.header_rect.bottom + 4
            max_preview_bottom = self.vu_rect.top - 4
            preview_h = max(24, max_preview_bottom - preview_y)
            preview_w = max(24, self.rect.width - 2 * pad)

            self.preview_rect = pygame.Rect(
                self.rect.left + pad,
                preview_y,
                preview_w,
                preview_h,
            )

            self.graph_rect = pygame.Rect(0, 0, 0, 0)
            self.q_slider_rect = pygame.Rect(0, 0, 0, 0)
            self.type_button_rects = []
            self.order_button_rects = []
            self.reset_button_rect = pygame.Rect(0, 0, 0, 0)
            self.reset_q_button_rect = pygame.Rect(0, 0, 0, 0)
            self.reset_default_button_rect = pygame.Rect(0, 0, 0, 0)

    def get_bands(self):
        """Use the per-frame snapshot when drawing, otherwise read state."""
        if self._frame_bands is not None:
            return self._frame_bands
        _, bands = self.eq_state.get_snapshot()
        return bands

    def _selected_band_supports_order(self):
        """Only high-pass and low-pass bands expose slope/order buttons."""
        if self.selected_band is None:
            return False
        bands = self.get_bands()
        if self.selected_band >= len(bands):
            return False
        return bands[self.selected_band].get("type", "bell") in ("hp", "highpass", "lp", "lowpass")

    def _q_to_slider_x(self, q):
        """Map Q value to slider position."""
        q_min, q_max = 0.2, 10.0
        t = (float(q) - q_min) / (q_max - q_min)
        t = max(0.0, min(1.0, t))
        return self.q_slider_rect.left + t * self.q_slider_rect.width

    def _slider_x_to_q(self, x):
        """Map slider position back to Q value."""
        if self.q_slider_rect.width <= 0:
            return 1.0
        q_min, q_max = 0.2, 10.0
        t = (x - self.q_slider_rect.left) / self.q_slider_rect.width
        t = max(0.0, min(1.0, t))
        return q_min + t * (q_max - q_min)

    def _freq_to_x(self, freq, rect):
        """Convert logarithmic frequency into screen x coordinate."""
        log_f = math.log10(max(self.min_freq, min(self.max_freq, float(freq))))
        t = (log_f - self.log_min_freq) / self.log_freq_span
        return rect.left + t * rect.width

    def _x_to_freq(self, x, rect):
        """Convert graph x coordinate back into frequency."""
        if rect.width <= 0:
            return 1000.0
        t = (x - rect.left) / rect.width
        t = max(0.0, min(1.0, t))
        return 10 ** (self.log_min_freq + t * self.log_freq_span)

    def _gain_to_y(self, gain_db, rect):
        """Convert gain in dB into screen y coordinate."""
        t = (float(gain_db) - self.min_gain) / (self.max_gain - self.min_gain)
        t = max(0.0, min(1.0, t))
        return rect.bottom - t * rect.height

    def _y_to_gain(self, y, rect):
        """Convert graph y coordinate back into gain in dB."""
        if rect.height <= 0:
            return 0.0
        t = (rect.bottom - y) / rect.height
        t = max(0.0, min(1.0, t))
        return self.min_gain + t * (self.max_gain - self.min_gain)

    def _band_pos(self, band, rect):
        """Return the draggable point location for one EQ band."""
        return self._freq_to_x(band["freq"], rect), self._gain_to_y(band["gain_db"], rect)

    def _find_band_near(self, mx, my, radius=14):
        """Find a band handle close enough to the mouse pointer."""
        bands = self.get_bands()
        for i, band in enumerate(bands):
            bx, by = self._band_pos(band, self.graph_rect)
            if math.hypot(mx - bx, my - by) <= radius:
                return i
        return None

    def butterworth_section_qs(self, order):
        """Forward to the DSP helper so GUI and audio math match."""
        return biquad_butterworth_section_qs(order)

    def _compute_curve_points(self, rect):
        """Build screen points for the combined EQ response curve."""
        if rect.width <= 0 or rect.height <= 0:
            return []

        bands = self.get_bands()
        fs = float(getattr(self.audio_engine, "fs", 48000.0))
        sos_rows = []

        for band in bands:
            # Build SOS rows for each enabled band, matching the C engine types.
            if not band.get("enabled", True):
                continue

            curve_type = band.get("type", "bell")
            f0 = float(band.get("freq", 1000.0))
            q = max(0.1, float(band.get("q", 1.0)))
            gain_db = float(band.get("gain_db", 0.0))
            order = int(band.get("order", 2))

            if curve_type == "bell":
                sos_rows.append(make_peaking_eq(fs, f0, q, gain_db).sos[0])

            elif curve_type in ("lp", "lowpass"):
                for section_q in self.butterworth_section_qs(order):
                    sos_rows.append(make_lowpass(fs, f0, q=section_q).sos[0])

            elif curve_type in ("hp", "highpass"):
                for section_q in self.butterworth_section_qs(order):
                    sos_rows.append(make_highpass(fs, f0, q=section_q).sos[0])

            elif curve_type == "low_shelf":
                sos_rows.append(make_low_shelf(fs, f0, gain_db, slope=1.0).sos[0])

            elif curve_type == "high_shelf":
                sos_rows.append(make_high_shelf(fs, f0, gain_db, slope=1.0).sos[0])

        if sos_rows:
            # sosfreqz evaluates the complete cascaded response for drawing.
            sos = np.asarray(sos_rows, dtype=np.float32)
            w = 2.0 * np.pi * self.curve_freqs / fs
            _, h = sosfreqz(sos, worN=w)
            total_gain_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
        else:
            total_gain_db = np.zeros_like(self.curve_freqs)

        points = []
        step = 3 if self.expanded else 6
        for f, g in zip(self.curve_freqs[::step], total_gain_db[::step]):
            points.append((self._freq_to_x(f, rect), self._gain_to_y(g, rect)))
        return points

    def _meter_level_to_height(self, db_value, height):
        """Scale a dBFS value to a meter height."""
        floor = -60.0
        t = (float(db_value) - floor) / (0.0 - floor)
        t = max(0.0, min(1.0, t))
        return int(height * t)

    def _meter_color(self, db_value):
        """Choose green/yellow/red by peak level."""
        if db_value >= -3.0:
            return (236, 76, 76)
        if db_value >= -12.0:
            return (238, 188, 78)
        return (90, 226, 130)

    def _meter_segment_color(self, t):
        """Choose segment color by normalized meter position."""
        if t >= 0.92:
            return (240, 70, 70)
        if t >= 0.76:
            return (234, 186, 72)
        return (88, 218, 126)
    

    def _format_freq_label(self, f):
        """Shorten frequency labels so they fit the graph."""
        if f >= 1000:
            val = f / 1000.0
            if abs(val - round(val)) < 1e-6:
                return f"{int(round(val))}k"
            return f"{val:.1f}k"
        return f"{int(f)}"

    def _format_db_label(self, db):
        """Add + sign to positive gain labels."""
        if db > 0:
            return f"+{int(db)}"
        return f"{int(db)}"

    def _draw_meter(self):
        """Draw either the compact horizontal meter or full vertical meter."""
        rect = self.vu_rect
        if rect.width <= 0 or rect.height <= 0:
            return

        pygame.draw.rect(self.screen, (12, 26, 36), rect)
        pygame.draw.rect(self.screen, self.axis_color, rect, 1)

        peak = float(self.meter.get("peak_dbfs", -120.0))
        hold = float(self.meter.get("peak_hold_dbfs", peak))
        clipped = bool(self.meter.get("clipped", False))

        hold_color = (255, 98, 98) if (clipped or hold >= -3.0) else (240, 196, 88)
        fill_color = (245, 82, 82) if clipped else self._meter_color(peak)

        if self.meter_horizontal:
            # Compact previews use segmented horizontal meters.
            total_segments = 18
            gap = 1

            inner_left = rect.left + 1
            inner_top = rect.top + 1
            inner_h = rect.height - 2
            inner_w = rect.width - 2

            seg_w = max(1, (inner_w - gap * (total_segments - 1)) // total_segments)
            bar_w = total_segments * seg_w + (total_segments - 1) * gap

            level_frac = max(0.0, min(1.0, (peak + 60.0) / 60.0))
            active_segments = int(round(level_frac * total_segments))

            x = inner_left
            for i in range(total_segments):
                t = (i + 1) / total_segments
                color = self._meter_segment_color(t) if i < active_segments else (38, 56, 68)
                seg_rect = pygame.Rect(x, inner_top, seg_w, inner_h)
                pygame.draw.rect(self.screen, color, seg_rect)
                x += seg_w + gap

            hold_frac = max(0.0, min(1.0, (hold + 60.0) / 60.0))
            hold_x = inner_left + int((bar_w - 1) * hold_frac)
            pygame.draw.line(self.screen, hold_color, (hold_x, rect.top - 1), (hold_x, rect.bottom + 1), 2)

            # Clip box to the right of the shortened mini meter.
            clip_rect = self.mini_clip_rect
            if clip_rect.width > 0 and clip_rect.height > 0:
                clip_fill = (240, 70, 70) if clipped else (52, 62, 72)
                pygame.draw.rect(self.screen, clip_fill, clip_rect)
                pygame.draw.rect(self.screen, self.axis_color, clip_rect, 1)
        
        else:
            # Expanded view uses a console-style vertical meter.
            total_segments = 24
            gap = 1
            seg_h = max(1, (rect.height - gap * (total_segments - 1) - 2) // total_segments)
            level_frac = max(0.0, min(1.0, (peak + 60.0) / 60.0))
            active_segments = int(round(level_frac * total_segments))

            y = rect.bottom - 1 - seg_h
            for i in range(total_segments):
                t = (i + 1) / total_segments
                color = self._meter_segment_color(t) if i < active_segments else (36, 54, 66)
                seg_rect = pygame.Rect(rect.left + 1, y, rect.width - 2, seg_h)
                pygame.draw.rect(self.screen, color, seg_rect)
                y -= seg_h + gap

            hold_frac = max(0.0, min(1.0, (hold + 60.0) / 60.0))
            hold_y = rect.bottom - int((rect.height - 2) * hold_frac)
            pygame.draw.line(self.screen, hold_color, (rect.left - 1, hold_y), (rect.right + 1, hold_y), 2)

            # Clip light above meter.
            clip_rect = pygame.Rect(rect.centerx - 5, rect.top - 16, 10, 10)
            clip_fill = (240, 70, 70) if clipped else (56, 66, 74)
            pygame.draw.rect(self.screen, clip_fill, clip_rect)
            pygame.draw.rect(self.screen, self.axis_color, clip_rect, 1)

            clip_text = self.tiny_font.render("CLP", True, self.text_color)
            self.screen.blit(
                clip_text,
                (rect.centerx - clip_text.get_width() // 2, clip_rect.top - clip_text.get_height() - 2),
            )

            db_text = self.tiny_font.render("dB", True, self.muted_text)
            value_text = self.small_font.render(f"{peak:5.1f}", True, fill_color)
            value_x = rect.centerx - value_text.get_width() // 2
            value_y = rect.bottom + 16
            self.screen.blit(value_text, (value_x, value_y))
            self.screen.blit(db_text, (rect.centerx - db_text.get_width() // 2, rect.bottom + 2))

    def _draw_header(self):
        """Draw the channel label at a size that fits the tile state."""
        if self.expanded:
            title = self.font.render(self.label, True, self.text_color)
            x = self.graph_rect.centerx - title.get_width() // 2
            self.screen.blit(title, (x, self.header_rect.top))
        else:
            title = self.tiny_font.render(self.label, True, self.text_color)
            x = self.header_rect.centerx - title.get_width() // 2
            self.screen.blit(title, (x, self.header_rect.top))

    def _draw_preview(self):
        """Draw minimized channel curve, pan position, and mini meter area."""
        rect = self.preview_rect
        if rect.width <= 0 or rect.height <= 0:
            return

        pygame.draw.rect(self.screen, (5, 16, 24), rect)
        pygame.draw.rect(self.screen, (70, 132, 170), rect, 1)

        pan_bar_h = 6
        pan_bar_w = max(24, rect.width - 20)
        pan_bar_rect = pygame.Rect(rect.left + 10, rect.bottom - pan_bar_h - 4, pan_bar_w, pan_bar_h)
        curve_rect = pygame.Rect(rect.left + 4, rect.top + 4, rect.width - 8, max(16, pan_bar_rect.top - rect.top - 8))

        # Zero line.
        y0 = int(self._gain_to_y(0.0, curve_rect))
        pygame.draw.line(self.screen, self.axis_color, (curve_rect.left, y0), (curve_rect.right, y0), 1)

        # Faint vertical splits for a more console-like thumbnail.
        # for frac in (0.25, 0.5, 0.75):
        #     x = rect.left + int(rect.width * frac)
        #     pygame.draw.line(self.screen, self.grid_color, (x, rect.top), (x, rect.bottom), 1)

        points = self._compute_curve_points(curve_rect)
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.preview_curve_color, False, points, 2)

        pan_value = self._get_pan_value()
        self._draw_pan_bar(pan_bar_rect, pan_value)

    def _get_pan_value(self):
        """Read pan from the engine, falling back to center on errors."""
        if hasattr(self.audio_engine, "get_channel_dial_values"):
            try:
                values = self.audio_engine.get_channel_dial_values()
                if self.track_index < len(values):
                    return max(-1.0, min(1.0, float(values[self.track_index])))
            except Exception:
                pass

        if hasattr(self.audio_engine, "track_dials"):
            try:
                values = self.audio_engine.track_dials
                if self.track_index < len(values):
                    return max(-1.0, min(1.0, float(values[self.track_index])))
            except Exception:
                pass

        return 0.0

    def _draw_pan_bar(self, rect, pan_value):
        """Draw a center-out pan bar from left through center to right."""
        if rect.width <= 0 or rect.height <= 0:
            return

        pan_value = max(-1.0, min(1.0, float(pan_value)))
        center_x = rect.centerx
        fill_half = max(1, rect.width // 2 - 2)
        line_x = center_x + int(round(pan_value * fill_half))

        pygame.draw.rect(self.screen, self.pan_track_color, rect, border_radius=3)
        pygame.draw.rect(self.screen, self.border_color, rect, 1, border_radius=3)

        center_w = max(2, rect.width // 30)
        center_rect = pygame.Rect(center_x - center_w // 2, rect.top, center_w, rect.height)
        pygame.draw.rect(self.screen, self.pan_center_color, center_rect, border_radius=2)

        if pan_value < 0.0:
            fill_rect = pygame.Rect(line_x, rect.top, max(0, center_x - line_x), rect.height)
            pygame.draw.rect(self.screen, self.pan_left_color, fill_rect, border_radius=3)
        elif pan_value > 0.0:
            fill_rect = pygame.Rect(center_x, rect.top, max(0, line_x - center_x), rect.height)
            pygame.draw.rect(self.screen, self.pan_right_color, fill_rect, border_radius=3)

        if abs(pan_value) < 1e-3:
            pygame.draw.line(self.screen, self.pan_center_glow, (center_x, rect.top - 1), (center_x, rect.bottom + 1), 3)
        else:
            color = self.pan_left_color if pan_value < 0.0 else self.pan_right_color
            pygame.draw.line(self.screen, color, (line_x, rect.top - 1), (line_x, rect.bottom + 1), 3)

        if abs(pan_value) < 1e-3:
            pygame.draw.line(self.screen, self.pan_center_glow, (center_x, rect.top - 2), (center_x, rect.bottom + 2), 1)

        if rect.height >= 10:
            l_label = self.tiny_font.render("L", True, self.text_color)
            c_label = self.tiny_font.render("C", True, self.text_color)
            r_label = self.tiny_font.render("R", True, self.text_color)
            self.screen.blit(l_label, (rect.left - 2, rect.bottom + 1))
            self.screen.blit(c_label, (rect.centerx - c_label.get_width() // 2, rect.bottom + 1))
            self.screen.blit(r_label, (rect.right - r_label.get_width() + 2, rect.bottom + 1))

    def _draw_graph(self):
        """Draw EQ grid, response curve, and draggable band handles."""
        rect = self.graph_rect
        if rect.width <= 0 or rect.height <= 0:
            return

        pygame.draw.rect(self.screen, (6, 18, 28), rect)
        pygame.draw.rect(self.screen, self.axis_color, rect, 1)

        freq_lines = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        gain_lines = [-18, -12, -6, 0, 6, 12, 18]

        for f in freq_lines:
            x = int(self._freq_to_x(f, rect))
            pygame.draw.line(self.screen, self.grid_color, (x, rect.top), (x, rect.bottom), 1)

            label = self.tiny_font.render(self._format_freq_label(f), True, self.muted_text)
            lx = x - label.get_width() // 2
            ly = rect.bottom + 3
            self.screen.blit(label, (lx, ly))

        for g in gain_lines:
            y = int(self._gain_to_y(g, rect))
            pygame.draw.line(self.screen, self.grid_color, (rect.left, y), (rect.right, y), 1)

            label = self.tiny_font.render(self._format_db_label(g), True, self.muted_text)
            lx = max(self.rect.left + 6, rect.left - label.get_width() - 5)
            ly = y - label.get_height() // 2
            self.screen.blit(label, (lx, ly))

        hz_label = self.tiny_font.render("Hz", True, self.axis_color)
        self.screen.blit(hz_label, (rect.right - hz_label.get_width(), rect.bottom + 16))
        db_label = self.tiny_font.render("dB", True, self.axis_color)
        db_x = max(self.rect.left + 6, rect.left - db_label.get_width() - 6)
        self.screen.blit(db_label, (db_x, rect.top - 25))

        if self.curve_dirty:
            # Recalculate only when a band changes or the tile moves/resizes.
            self.cached_curve_points = self._compute_curve_points(rect)
            self.curve_dirty = False

        if len(self.cached_curve_points) > 1:
            pygame.draw.lines(self.screen, self.curve_color, False, self.cached_curve_points, 3)

        bands = self.get_bands()
        for i, band in enumerate(bands):
            bx, by = self._band_pos(band, rect)
            color = self.band_colors[i % len(self.band_colors)]
            radius = 9 if i == self.selected_band else 7

            pygame.draw.circle(self.screen, color, (int(bx), int(by)), radius)
            pygame.draw.circle(self.screen, (242, 247, 250), (int(bx), int(by)), radius, 2)

            idx = self.tiny_font.render(str(i + 1), True, (18, 18, 18))
            self.screen.blit(idx, (int(bx) - 3, int(by) - 5))

    def _draw_controls(self):
        """Draw Q slider, filter type/order buttons, reset buttons, and pan."""
        if self.q_slider_rect.width <= 0:
            return

        bands = self.get_bands()
        current_band = bands[self.selected_band] if (self.selected_band is not None and self.selected_band < len(bands)) else None

        control_fill = (14, 34, 48)
        control_fill_active = (24, 104, 146)
        control_fill_disabled = (18, 24, 30)
        control_border = (82, 142, 176)
        control_border_active = self.border_active

        pan_value = self._get_pan_value()
        pan_label = self.tiny_font.render("PAN", True, self.text_color)
        pan_rect = pygame.Rect(self.graph_rect.right - 126, self.header_rect.top + 5, 116, 8)
        self.screen.blit(pan_label, (pan_rect.left - pan_label.get_width() - 6, pan_rect.top - 1))
        self._draw_pan_bar(pan_rect, pan_value)

        # Q row.
        q_label = self.small_font.render("Q", True, self.text_color)
        self.screen.blit(q_label, (self.q_slider_rect.left - 24, self.q_slider_rect.top - 7))

        pygame.draw.line(
            self.screen,
            self.axis_color,
            (self.q_slider_rect.left, self.q_slider_rect.centery),
            (self.q_slider_rect.right, self.q_slider_rect.centery),
            4,
        )

        if current_band is not None:
            knob_x = int(self._q_to_slider_x(float(current_band.get("q", 1.0))))
            pygame.draw.circle(
                self.screen,
                (242, 247, 250),
                (knob_x, self.q_slider_rect.centery),
                self.q_slider_knob_radius + 1,
            )
            q_text = self.tiny_font.render(f"{float(current_band.get('q', 1.0)):.2f}", True, self.text_color)
            self.screen.blit(q_text, (knob_x + 10, self.q_slider_rect.top + 8))

        if current_band is None:
            msg = self.tiny_font.render("Select band", True, self.muted_text)
            self.screen.blit(msg, (self.q_slider_rect.left, self.q_slider_rect.bottom + 14))
            return

        current_type = current_band.get("type", "bell")
        current_order = int(current_band.get("order", 2))
        show_order = current_type in ("hp", "highpass", "lp", "lowpass")

        # TYPE label.
        if self.type_button_rects:
            type_label = self.tiny_font.render("TYPE", True, self.muted_text)
            self.screen.blit(type_label, (self.type_button_rects[0][1].left, self.type_button_rects[0][1].top - 14))

        for label, rect in self.type_button_rects:
            band_type = label
            if label == "LS":
                band_type = "low_shelf"
            elif label == "HS":
                band_type = "high_shelf"

            active = band_type == current_type
            fill = control_fill_active if active else control_fill
            border = control_border_active if active else control_border

            pygame.draw.rect(self.screen, fill, rect, border_radius=4)
            pygame.draw.rect(self.screen, border, rect, 1, border_radius=4)

            text = self.tiny_font.render(label, True, self.text_color)
            self.screen.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))

        # ORDER label.
        if self.order_button_rects:
            order_label = self.tiny_font.render("ORDER", True, self.text_color if show_order else self.muted_text)
            self.screen.blit(order_label, (self.order_button_rects[0][1].left, self.order_button_rects[0][1].top - 14))

        for order, rect in self.order_button_rects:
            active = show_order and (order == current_order)
            fill = control_fill_active if active else (control_fill if show_order else control_fill_disabled)
            border = control_border_active if active else control_border
            text_color = self.text_color if show_order else self.muted_text

            pygame.draw.rect(self.screen, fill, rect, border_radius=4)
            pygame.draw.rect(self.screen, border, rect, 1, border_radius=4)

            text = self.tiny_font.render(str(order), True, text_color)
            self.screen.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))

        # Reset label and buttons.
        reset_label = self.tiny_font.render("RESET", True, self.muted_text)
        self.screen.blit(
            reset_label,
            (self.reset_default_button_rect.left, self.reset_default_button_rect.top - 16),
        )

        reset_fill = (44, 72, 94)
        
        # Default button.
        pygame.draw.rect(self.screen, reset_fill, self.reset_default_button_rect, border_radius=4)
        pygame.draw.rect(self.screen, control_border, self.reset_default_button_rect, 1, border_radius=4)
        default_text = self.tiny_font.render("Default", True, self.text_color)
        self.screen.blit(
            default_text,
            (self.reset_default_button_rect.centerx - default_text.get_width() // 2,
             self.reset_default_button_rect.centery - default_text.get_height() // 2),
        )

        # Gain button.
        pygame.draw.rect(self.screen, reset_fill, self.reset_button_rect, border_radius=4)
        pygame.draw.rect(self.screen, control_border, self.reset_button_rect, 1, border_radius=4)
        gain_text = self.tiny_font.render("Gain", True, self.text_color)
        self.screen.blit(
            gain_text,
            (self.reset_button_rect.centerx - gain_text.get_width() // 2,
             self.reset_button_rect.centery - gain_text.get_height() // 2),
        )

        # Q value button.
        pygame.draw.rect(self.screen, reset_fill, self.reset_q_button_rect, border_radius=4)
        pygame.draw.rect(self.screen, control_border, self.reset_q_button_rect, 1, border_radius=4)
        q_text = self.tiny_font.render("Q value", True, self.text_color)
        self.screen.blit(
            q_text,
            (self.reset_q_button_rect.centerx - q_text.get_width() // 2,
             self.reset_q_button_rect.centery - q_text.get_height() // 2),
        )

    def _request_rebuild(self):
        """Ask the audio engine to send current EQ state to the C runtime."""
        if hasattr(self.audio_engine, "request_rebuild"):
            self.audio_engine.request_rebuild()

    def _request_rebuild_throttled(self):
        """Limit rebuild traffic while a user drags a control."""
        now = time.time()
        if now - self.last_rebuild_time > 0.04:
            self._request_rebuild()
            self.last_rebuild_time = now

    def handle_event(self, event):
        """Handle mouse input for selecting, dragging, and editing bands."""
        action = {}
        if self.disabled:
            return action

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            if not self.rect.collidepoint(mx, my):
                return action

            if not self.expanded:
                # Minimized tiles notify the parent that they should expand.
                action["select_track"] = self.track_index
                return action

            band_index = self._find_band_near(mx, my)
            if band_index is not None:
                # Double-click a band to reset its gain and Q quickly.
                if getattr(event, "button", 1) == 1 and getattr(event, "clicks", 1) >= 2:
                    self.eq_state.update_band(band_index, gain_db=0.0, q=1.0)
                    self.selected_band = band_index
                    self.curve_dirty = True
                    self._request_rebuild()
                    return action
                self.selected_band = band_index
                self.dragging_band = True
                return action

            if self.selected_band is not None:
                bands = self.get_bands()
                if self.selected_band < len(bands):
                    knob_x = self._q_to_slider_x(float(bands[self.selected_band].get("q", 1.0)))
                    knob_y = self.q_slider_rect.centery
                    if math.hypot(mx - knob_x, my - knob_y) <= self.q_slider_knob_radius + 3:
                        self.dragging_q = True
                        return action
                    
            for label, rect in self.type_button_rects:
                # Type buttons switch the selected band's filter model.
                if rect.collidepoint(mx, my) and self.selected_band is not None:
                    new_type = label
                    if label == "LS":
                        new_type = "low_shelf"
                    elif label == "HS":
                        new_type = "high_shelf"

                    self.eq_state.update_band(self.selected_band, type=new_type)
                    self.curve_dirty = True
                    self._request_rebuild()
                    return action

            if self._selected_band_supports_order() and self.selected_band is not None:
                # Order buttons only apply to high-pass and low-pass filters.
                for order, rect in self.order_button_rects:
                    if rect.collidepoint(mx, my):
                        self.eq_state.update_band(self.selected_band, order=int(order))
                        self.curve_dirty = True
                        self._request_rebuild()
                        return action

            if self.reset_button_rect.collidepoint(mx, my) and self.selected_band is not None:
                self.eq_state.update_band(self.selected_band, gain_db=0.0)
                self.curve_dirty = True
                self._request_rebuild()
                return action

            if self.reset_q_button_rect.collidepoint(mx, my) and self.selected_band is not None:
                self.eq_state.update_band(self.selected_band, q=1.0)
                self.curve_dirty = True
                self._request_rebuild()
                return action

            if self.reset_default_button_rect.collidepoint(mx, my) and self.selected_band is not None:
                _, bands = self.eq_state.get_snapshot()
                original_freq = bands[self.selected_band].get("original_freq", bands[self.selected_band]["freq"])
                self.eq_state.update_band(self.selected_band, freq=original_freq, gain_db=0.0, q=1.0, type="bell")
                self.curve_dirty = True
                self._request_rebuild()
                return action


        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging_band or self.dragging_q:
                self.dragging_band = False
                self.dragging_q = False
                self._request_rebuild()

        elif event.type == pygame.MOUSEMOTION:
            if not self.expanded:
                return action

            if self.dragging_band and self.selected_band is not None:
                # Dragging a handle changes frequency and gain together.
                mx, my = event.pos
                if self.graph_rect.collidepoint(mx, my):
                    self.eq_state.update_band(
                        self.selected_band,
                        freq=self._x_to_freq(mx, self.graph_rect),
                        gain_db=self._y_to_gain(my, self.graph_rect),
                    )
                    self.curve_dirty = True
                    self._request_rebuild_throttled()

            if self.dragging_q and self.selected_band is not None:
                # Q dragging changes bandwidth/resonance without moving the handle.
                mx, _ = event.pos
                self.eq_state.update_band(self.selected_band, q=self._slider_x_to_q(mx))
                self.curve_dirty = True
                self._request_rebuild_throttled()

        return action

    def update(self, dt):
        """Reserved for animations; currently only validates expanded state."""
        if not self.expanded:
            return
        if dt < 0:
            return

    def draw(self):
        """Draw the tile in disabled, preview, or expanded mode."""
        if self.disabled:
            pygame.draw.rect(self.screen, (6, 10, 14), self.rect, border_radius=6)
            pygame.draw.rect(self.screen, (30, 40, 48), self.rect, 2, border_radius=6)

            label = self.tiny_font.render("DISCONNECTED", True, self.muted_text)
            self.screen.blit(
                label,
                (
                    self.rect.centerx - label.get_width() // 2,
                    self.rect.centery - label.get_height() // 2,
                ),
            )
            return

        tile_color = self.bg_active if self.expanded else self.bg_color
        border_color = self.border_active if self.expanded else self.border_color

        pygame.draw.rect(self.screen, tile_color, self.rect, border_radius=6)
        pygame.draw.rect(self.screen, border_color, self.rect, 2, border_radius=6)

        # One snapshot per frame keeps graph/handles internally consistent.
        _, self._frame_bands = self.eq_state.get_snapshot()

        self._draw_header()
        self._draw_meter()
        if self.expanded:
            self._draw_graph()
            self._draw_controls()
        else:
            self._draw_preview()
            if self.preview_dim_alpha > 0:
                dim_overlay = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
                dim_overlay.fill((0, 0, 0, self.preview_dim_alpha))
                self.screen.blit(dim_overlay, (self.rect.left, self.rect.top))
            if self.preview_tint_alpha > 0:
                tint_overlay = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
                tint_overlay.fill((176, 42, 42, self.preview_tint_alpha))
                self.screen.blit(tint_overlay, (self.rect.left, self.rect.top))

        self._frame_bands = None

class EQFourChannelView:
    """Parent view that manages 4 EQChannelTile instances and a master meter."""

    def __init__(self, screen, eq_state, audio_engine):
        # The parent owns channel selection, mute state, and master layout.
        self.screen = screen
        self.audio_engine = audio_engine
        self.running = True
        self.width = screen.get_width()
        self.height = screen.get_height()

        self.bg_color = (4, 9, 14)
        self.master_bg = (10, 22, 32)
        self.master_border = (82, 162, 204)
        self.text_color = (228, 238, 245)
        self.small_font = pygame.font.SysFont("DejaVu Sans", 14)
        self.tiny_font = pygame.font.SysFont("DejaVu Sans", 11)

        self.track_count = 4
        self.active_track_count = self._get_active_track_count()
        self.selected_track = 0
        self.master_rect = pygame.Rect(0, 0, 0, 0)
        self.quit_button_rect = pygame.Rect(0, 0, 0, 0)
        self.track_rects = []
        self.left_preview_rects = []
        self.left_mute_button_rects = []
        self.track_muted = [False for _ in range(self.track_count)]
        self.track_unmuted_trim = [1.0 for _ in range(self.track_count)]

        self.eq_states = self._resolve_eq_states(eq_state)
        self.track_labels = [f"Channel {i}" for i in range(self.track_count)]

        self.tiles = []
        for i in range(self.track_count):
            tile = EQChannelTile(
                screen=screen,
                eq_state=self.eq_states[i],
                audio_engine=audio_engine,
                track_index=i,
                label=self.track_labels[i],
            )
            self.tiles.append(tile)

        self._track_meter_cache = [
            {"peak_dbfs": -120.0, "peak_hold_dbfs": -120.0, "clipped": False}
            for _ in range(self.track_count)
        ]

        if hasattr(self.audio_engine, "tracks"):
            tracks = getattr(self.audio_engine, "tracks", [])
            for i in range(min(self.track_count, len(tracks))):
                if hasattr(tracks[i], "output_trim"):
                    self.track_unmuted_trim[i] = float(getattr(tracks[i], "output_trim", 1.0))

        self._layout_dirty = True
        self._sync_active_track_from_engine()

    def _get_active_track_count(self):
        """Detect how many tracks the engine is actually running."""
        engine_tracks = getattr(self.audio_engine, "num_tracks", None)
        if engine_tracks is None:
            engine_tracks = len(getattr(self.audio_engine, "eq_states", []))
        if engine_tracks in (None, 0):
            engine_tracks = len(getattr(self.audio_engine, "tracks", []))
        try:
            engine_tracks = int(engine_tracks)
        except Exception:
            engine_tracks = self.track_count
        return max(1, min(self.track_count, engine_tracks))

    def _is_track_active(self, track_index):
        """Return whether a track exists in the current engine setup."""
        return 0 <= int(track_index) < self.active_track_count

    def _set_engine_track_mute(self, track_index, muted):
        """Call whichever mute API the current audio engine supports."""
        idx = int(track_index)
        muted = bool(muted)

        if hasattr(self.audio_engine, "set_track_muted"):
            self.audio_engine.set_track_muted(idx, muted)
            return
        if hasattr(self.audio_engine, "set_track_mute"):
            self.audio_engine.set_track_mute(idx, muted)
            return
        if muted and hasattr(self.audio_engine, "mute_track"):
            self.audio_engine.mute_track(idx)
            return
        if (not muted) and hasattr(self.audio_engine, "unmute_track"):
            self.audio_engine.unmute_track(idx)
            return

        tracks = getattr(self.audio_engine, "tracks", None)
        if tracks is not None and 0 <= idx < len(tracks):
            track = tracks[idx]
            if hasattr(track, "output_trim"):
                if muted:
                    setattr(track, "output_trim", 0.0)
                else:
                    setattr(track, "output_trim", float(self.track_unmuted_trim[idx]))

    def _set_track_muted(self, track_index, muted):
        """Update local mute state and mirror it to the audio engine."""
        idx = max(0, min(int(track_index), self.track_count - 1))
        if not self._is_track_active(idx):
            return
        muted = bool(muted)
        self.track_muted[idx] = muted
        self._set_engine_track_mute(idx, muted)
        self._update_preview_styles()

    def _toggle_track_muted(self, track_index):
        """Invert the mute state for one active track."""
        idx = max(0, min(int(track_index), self.track_count - 1))
        self._set_track_muted(idx, not self.track_muted[idx])

    def _update_preview_styles(self):
        """Dim inactive previews and tint muted tracks."""
        for i, tile in enumerate(self.tiles):
            is_active = self._is_track_active(i)
            tile.set_disabled(not is_active)
            tile.set_preview_style(
                dimmed=(i != self.selected_track) or (not is_active),
                muted_tint=self.track_muted[i] if is_active else True,
            )

    def _resolve_eq_states(self, eq_state):
        """Create four independent EQ state objects for the tiles."""
        if hasattr(self.audio_engine, "eq_states"):
            states = list(self.audio_engine.eq_states)
        elif isinstance(eq_state, (list, tuple)):
            states = list(eq_state)
        else:
            states = [eq_state]

        if not states:
            raise ValueError("No EQ states provided")

        # If a shared object is reused across channels, clone it so each channel is independent.
        deduped = []
        seen_ids = set()
        for state in states:
            if id(state) in seen_ids:
                state = self._clone_eq_state(state)
            seen_ids.add(id(state))
            deduped.append(state)
        states = deduped

        while len(states) < self.track_count:
            states.append(self._clone_eq_state(states[-1]))

        states = states[: self.track_count]

        if hasattr(self.audio_engine, "eq_states"):
            self.audio_engine.eq_states = list(states)

        return states

    def _clone_eq_state(self, source_state):
        """Duplicate an EQState without sharing its band dictionaries."""
        clone = EQState()
        _, bands = source_state.get_snapshot()
        clone.bands = [band.copy() for band in bands]
        clone.version = 0
        clone.min_freq = float(getattr(source_state, "min_freq", clone.min_freq))
        clone.max_freq = float(getattr(source_state, "max_freq", clone.max_freq))
        clone.min_gain = float(getattr(source_state, "min_gain", clone.min_gain))
        clone.max_gain = float(getattr(source_state, "max_gain", clone.max_gain))
        return clone

    def _sync_active_track_from_engine(self):
        """Start the GUI on the same selected track as the engine."""
        idx = 0
        if hasattr(self.audio_engine, "get_active_track_index"):
            idx = int(self.audio_engine.get_active_track_index())
        self.set_active_track(idx)

    def set_active_track(self, track_index):
        """Expand one track and collapse the others into previews."""
        track_index = max(0, min(int(track_index), self.active_track_count - 1))
        if hasattr(self.audio_engine, "set_active_track"):
            track_index = int(self.audio_engine.set_active_track(track_index))
            track_index = max(0, min(track_index, self.active_track_count - 1))

        self.selected_track = track_index
        for i, tile in enumerate(self.tiles):
            tile.set_expanded(i == self.selected_track)
        self._update_preview_styles()
        self._layout_dirty = True

    def _rebuild_layout(self):
        """Compute the full four-channel layout for the current screen size."""
        self.width = self.screen.get_width()
        self.height = self.screen.get_height()

        margin = 6
        top = margin
        height = self.height - margin * 2
        master_w = 44
        quit_h = 30
        quit_gap = 8

        master_x = self.width - margin - master_w
        self.quit_button_rect = pygame.Rect(master_x, top, master_w, quit_h)

        master_top = self.quit_button_rect.bottom + quit_gap
        master_h = max(80, height - quit_h - quit_gap)
        self.master_rect = pygame.Rect(master_x, master_top, master_w, master_h)

        channel_area_w = self.master_rect.left - margin * 2
        mini_gap = 8
        mini_w = 92
        mini_col_w = mini_w

        mini_h = (height - mini_gap * (self.track_count - 1)) // self.track_count
        mini_h = max(78, mini_h)
        max_mini_h = (height - mini_gap * (self.track_count - 1)) // self.track_count
        mini_h = min(mini_h, max_mini_h)

        selected_x = margin + mini_col_w + margin
        selected_w = channel_area_w - mini_col_w - margin
        if selected_w < int(channel_area_w * 0.62):
            mini_w = max(52, channel_area_w - int(channel_area_w * 0.62) - margin)
            mini_col_w = mini_w
            selected_x = margin + mini_col_w + margin
            selected_w = channel_area_w - mini_col_w - margin

        self.track_rects = [pygame.Rect(0, 0, 0, 0) for _ in range(self.track_count)]
        self.left_preview_rects = [pygame.Rect(0, 0, 0, 0) for _ in range(self.track_count)]
        self.left_mute_button_rects = [pygame.Rect(0, 0, 0, 0) for _ in range(self.track_count)]

        # Expanded channel gets the full-height editor region.
        expanded_rect = pygame.Rect(selected_x, top, selected_w, height)
        self.track_rects[self.selected_track] = expanded_rect
        self.tiles[self.selected_track].set_bounds(expanded_rect, expanded=True)
        self.tiles[self.selected_track].label = self.track_labels[self.selected_track]

        # Keep all four previews visible on the left at all times.
        used_h = self.track_count * mini_h + (self.track_count - 1) * mini_gap
        y0 = top + max(0, (height - used_h) // 2)
        for track_idx in range(self.track_count):
            y = y0 + track_idx * (mini_h + mini_gap)
            rect = pygame.Rect(margin, y, mini_w, mini_h)
            self.left_preview_rects[track_idx] = rect

            mute_size = 16
            mute_rect = pygame.Rect(rect.right - mute_size - 5, rect.top + 4, mute_size, mute_size)
            self.left_mute_button_rects[track_idx] = mute_rect

            if track_idx != self.selected_track:
                self.track_rects[track_idx] = rect
                self.tiles[track_idx].set_bounds(rect, expanded=False)
                self.tiles[track_idx].label = self.track_labels[track_idx]

        self._layout_dirty = False

    def _draw_selected_left_preview(self):
        """Temporarily draw the selected tile as a left-column preview too."""
        idx = self.selected_track
        if idx < 0 or idx >= len(self.tiles) or idx >= len(self.left_preview_rects):
            return

        tile = self.tiles[idx]
        left_rect = pygame.Rect(self.left_preview_rects[idx])
        expanded_rect = pygame.Rect(self.track_rects[idx])

        tile.set_bounds(left_rect, expanded=False)
        tile.set_preview_style(dimmed=False, muted_tint=self.track_muted[idx])
        tile.label = self.track_labels[idx]
        tile.draw()

        tile.set_bounds(expanded_rect, expanded=True)
        tile.set_preview_style(dimmed=False, muted_tint=self.track_muted[idx])

    def _draw_left_mute_buttons(self):
        """Draw small mute toggles over the preview column."""
        for i, rect in enumerate(self.left_mute_button_rects):
            if not self._is_track_active(i):
                fill = (16, 20, 24)
                border = (30, 36, 42)
                pygame.draw.rect(self.screen, fill, rect, border_radius=3)
                pygame.draw.rect(self.screen, border, rect, 1, border_radius=3)
                continue

            muted = self.track_muted[i]
            fill = (164, 58, 58) if muted else (28, 52, 68)
            border = (242, 126, 126) if muted else (84, 138, 168)
            text_color = (252, 234, 234) if muted else (214, 232, 244)

            pygame.draw.rect(self.screen, fill, rect, border_radius=3)
            pygame.draw.rect(self.screen, border, rect, 1, border_radius=3)

            txt = self.tiny_font.render("M", True, text_color)
            self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    def _draw_quit_button(self):
        """Draw the touchscreen-friendly quit button."""
        rect = self.quit_button_rect
        pygame.draw.rect(self.screen, (132, 42, 42), rect, border_radius=5)
        pygame.draw.rect(self.screen, (224, 128, 128), rect, 1, border_radius=5)
        label = self.small_font.render("QUIT", True, (250, 236, 236))
        self.screen.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))

    def _get_master_meter(self):
        """Read master meter values from the audio engine."""
        if hasattr(self.audio_engine, "get_output_meter"):
            meter = self.audio_engine.get_output_meter()
            return {
                "peak_dbfs": float(meter.get("peak_dbfs", -120.0)),
                "peak_hold_dbfs": float(meter.get("peak_hold_dbfs", meter.get("peak_dbfs", -120.0))),
                "clipped": bool(meter.get("clipped", False)),
            }
        return {"peak_dbfs": -120.0, "peak_hold_dbfs": -120.0, "clipped": False}

    def _get_track_meter(self, track_index, dt):
        """Read per-track meter data, or synthesize a fallback from master."""
        if not self._is_track_active(track_index):
            return {
                "peak_dbfs": -120.0,
                "peak_hold_dbfs": -120.0,
                "clipped": False,
            }

        if hasattr(self.audio_engine, "get_track_output_meter"):
            meter = self.audio_engine.get_track_output_meter(track_index)
            return {
                "peak_dbfs": float(meter.get("peak_dbfs", -120.0)),
                "peak_hold_dbfs": float(meter.get("peak_hold_dbfs", meter.get("peak_dbfs", -120.0))),
                "clipped": bool(meter.get("clipped", False)),
            }

        master = self._get_master_meter()
        base_peak = float(master.get("peak_dbfs", -120.0))
        osc = math.sin(time.time() * (2.5 + 0.45 * track_index) + track_index * 0.9)
        target_peak = max(-60.0, min(6.0, base_peak + (track_index - 1.5) * 1.1 + osc * 0.9))

        cache = self._track_meter_cache[track_index]
        prev = float(cache.get("peak_dbfs", -120.0))
        attack = 0.55
        release = max(0.02, min(0.22, 0.12 + dt * 0.4))
        alpha = attack if target_peak > prev else release
        peak = prev + alpha * (target_peak - prev)

        hold = float(cache.get("peak_hold_dbfs", -120.0))
        if peak >= hold:
            hold = peak
        else:
            hold = max(peak, hold - 28.0 * dt)

        clipped = bool(master.get("clipped", False) and track_index == self.selected_track and peak > -0.7)
        cache.update({"peak_dbfs": peak, "peak_hold_dbfs": hold, "clipped": clipped})
        return dict(cache)

    def _meter_frac(self, db_value):
        """Normalize dBFS values to the 0..1 meter range."""
        floor = -60.0
        t = (float(db_value) - floor) / (0.0 - floor)
        return max(0.0, min(1.0, t))

    def _meter_color(self, db_value):
        """Choose master meter text color from peak level."""
        if db_value >= -3.0:
            return (236, 76, 76)
        if db_value >= -12.0:
            return (238, 188, 78)
        return (90, 226, 130)

    def _draw_master_meter(self):
        """Draw the master output meter on the right edge."""
        pygame.draw.rect(self.screen, self.master_bg, self.master_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.master_border, self.master_rect, 2, border_radius=6)

        title = self.tiny_font.render("MAS", True, self.text_color)
        self.screen.blit(title, (self.master_rect.centerx - title.get_width() // 2, self.master_rect.top + 6))

        meter = self._get_master_meter()
        peak = float(meter.get("peak_dbfs", -120.0))
        hold = float(meter.get("peak_hold_dbfs", peak))
        clipped = bool(meter.get("clipped", False))

        bar = pygame.Rect(
            self.master_rect.centerx - 6,
            self.master_rect.top + 52,
            12,
            self.master_rect.height - 92,
        )

        pygame.draw.rect(self.screen, (18, 34, 45), bar)
        pygame.draw.rect(self.screen, (88, 136, 164), bar, 1)

        total_segments = 34
        gap = 1
        seg_h = max(1, (bar.height - gap * (total_segments - 1) - 2) // total_segments)
        frac = self._meter_frac(peak)
        active_segments = int(round(frac * total_segments))

        y = bar.bottom - 1 - seg_h
        for i in range(total_segments):
            t = (i + 1) / total_segments
            if i < active_segments:
                if t >= 0.92:
                    color = (240, 70, 70)
                elif t >= 0.76:
                    color = (234, 186, 72)
                else:
                    color = (88, 218, 126)
            else:
                color = (38, 56, 68)
            seg_rect = pygame.Rect(bar.left + 1, y, bar.width - 2, seg_h)
            pygame.draw.rect(self.screen, color, seg_rect)
            y -= seg_h + gap

        hold_y = bar.bottom - int((bar.height - 2) * self._meter_frac(hold))
        hold_color = (255, 98, 98) if (clipped or hold >= -3.0) else (240, 196, 88)
        pygame.draw.line(self.screen, hold_color, (bar.left - 2, hold_y), (bar.right + 2, hold_y), 2)

        # Clip light.
        clip_rect = pygame.Rect(self.master_rect.centerx - 5, self.master_rect.top + 24, 10, 10)
        clip_fill = (240, 70, 70) if clipped else (58, 66, 74)
        pygame.draw.rect(self.screen, clip_fill, clip_rect)
        pygame.draw.rect(self.screen, (96, 132, 156), clip_rect, 1)

        clip_text = self.tiny_font.render("CLP", True, self.text_color)
        self.screen.blit(clip_text, (self.master_rect.centerx - clip_text.get_width() // 2, clip_rect.bottom + 2))

        # Scale labels.
        for db in (0, -6, -12, -24, -48):
            ty = bar.bottom - int((bar.height - 2) * max(0.0, min(1.0, (db + 60.0) / 60.0)))
            txt = self.tiny_font.render(str(db), True, (118, 146, 165))
            self.screen.blit(txt, (self.master_rect.left + 3, ty - txt.get_height() // 2))

        db_text = self.tiny_font.render("dB", True, (118, 146, 165))
        self.screen.blit(db_text, (self.master_rect.left + 4, self.master_rect.bottom - 34))

        value_text = self.small_font.render(f"{peak:5.1f}", True, self._meter_color(peak))
        self.screen.blit(value_text, (self.master_rect.left + 4, self.master_rect.bottom - 18))

    def handle_event(self, event):
        """Handle parent-level clicks before forwarding to the selected tile."""
        if self._layout_dirty:
            self._rebuild_layout()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos

            if self.quit_button_rect.collidepoint(mx, my):
                self.running = False
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return

            for i, rect in enumerate(self.left_mute_button_rects):
                hit_rect = rect.inflate(18, 18)
                if self._is_track_active(i) and hit_rect.collidepoint(mx, my):
                    self._toggle_track_muted(i)
                    return

            for i, rect in enumerate(self.left_preview_rects):
                if self._is_track_active(i) and rect.collidepoint(mx, my):
                    self.set_active_track(i)
                    return

        self.tiles[self.selected_track].handle_event(event)

    def update(self, dt):
        """Refresh meters and update all channel tiles."""
        if self._layout_dirty:
            self._rebuild_layout()

        for i, tile in enumerate(self.tiles):
            tile.set_meter(self._get_track_meter(i, dt))
            tile.update(dt)

    def draw(self):
        """Draw the full mixer screen in back-to-front order."""
        if self._layout_dirty:
            self._rebuild_layout()

        self.screen.fill(self.bg_color)
        self._draw_selected_left_preview()
        for i, tile in enumerate(self.tiles):
            if i == self.selected_track:
                continue
            tile.draw()
        self._draw_left_mute_buttons()
        self._draw_quit_button()
        self.tiles[self.selected_track].draw()
        self._draw_master_meter()


class EQLiveGui(EQFourChannelView):
    """Top-level GUI class used by main loops.

    Keeps the same constructor style as previous GUIs:
        EQLiveGui(screen, eq_state, audio_engine)
    """

    def __init__(self, screen, eq_state, audio_engine):
        super().__init__(screen, eq_state, audio_engine)

    def handle_event(self, event):
        # Keep common keyboard behavior from earlier GUI iterations.
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                idx = event.key - pygame.K_1
                if idx < self.track_count:
                    self.set_active_track(idx)
                return

            if event.key == pygame.K_r and hasattr(self.audio_engine, "command_q"):
                self.audio_engine.command_q.put("restart")
                return

            if event.key == pygame.K_SPACE and hasattr(self.audio_engine, "command_q"):
                self.audio_engine.command_q.put("toggle_play")
                return

        super().handle_event(event)
