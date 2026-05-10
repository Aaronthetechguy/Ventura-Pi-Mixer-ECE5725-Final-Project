"""Serial control bridge for hardware faders and encoders.

This file reads simple text messages from a microcontroller, stores the most
recent fader/pan values, and pushes changes into the live audio engine.
"""

import threading
import time

try:
    import serial  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - runtime environment dependent
    serial = None


class SerialControlBridge:
    """Simple non-blocking serial receiver for fader/dial control.

    Incoming protocol:
      fader1:0.75
      dial1:2.5

    Outgoing protocol:
      dial1:2.5
      dial2:0.0
      ...
    """

    def __init__(self, port="/dev/ttyACM0", baudrate=115200, channel_count=4):
        self.port = port
        self.baudrate = int(baudrate)
        self.channel_count = max(1, int(channel_count))

        self.fader_values = [1.0] * self.channel_count
        self.dial_values = [0.0] * self.channel_count

        self._engine = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._ser = None
        self._last_pushed_faders = None
        self._last_pushed_dials = None
        self.debug = False

    def bind_engine(self, engine):
        """Attach the audio engine that should receive hardware changes."""
        self._engine = engine

    def _debug_print(self, message):
        """Print only when the bridge is in debug mode."""
        if self.debug:
            print(message)

    def open(self):
        """Open the serial port once and prepare it for line-based reads."""
        if self._ser is not None:
            return
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install with: pip install pyserial")
        # Use a short blocking timeout so each Arduino update is handled as it arrives
        # without a busy-spin loop.
        self._ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=0.1, write_timeout=0)
        self._ser.reset_input_buffer()

    def close(self):
        """Stop polling and close the serial port."""
        self.stop()
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def start(self):
        """Start the background serial polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Ask the polling thread to exit and wait briefly for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    def _run_loop(self):
        """Continuously poll serial input until stop() is called."""
        while not self._stop.is_set():
            self.poll_once()

    def poll_once(self):
        """Read pending serial lines, then push changed values to the engine."""
        self._read_available_lines()
        self._push_to_engine()

    def _read_available_lines(self):
        """Drain all currently available serial lines."""
        if self._ser is None:
            return

        while not self._stop.is_set():
            raw = self._ser.readline()
            if not raw:
                break

            line = raw.decode("utf-8", errors="replace").strip()
            self._apply_line(line)

    def _apply_line(self, line):
        """Parse one fader/encoder line and update the local control cache."""
        if not line or ":" not in line:
            return

        try:
            name, value_str = line.split(":", 1)
            value = float(value_str)
        except ValueError:
            return

        key = name.strip().lower()
        idx = self._parse_channel_index(key)
        if idx is None or idx >= self.channel_count:
            return

        # Clamp raw hardware values before exposing them to the engine.
        with self._lock:
            if key.startswith("fader"):
                self.fader_values[idx] = max(0.0, min(10.0, float(value)))
                self._debug_print(f"rx fader{idx}: {self.fader_values[idx]:.3f}")
            elif key.startswith("encoder"):
                self.dial_values[idx] = max(-1.0, min(1.0, float(value)))
                self._debug_print(f"rx encoder{idx}: {self.dial_values[idx]:.3f}")

    def _parse_channel_index(self, key):
        """Extract a channel number from names like fader1 or encoder2."""
        digits = ""
        for ch in key:
            if ch.isdigit():
                digits += ch
        if not digits:
            return None

        # Prefer 1-based sender naming (encoder1 -> channel 0, encoder2 -> channel 1).
        # Still accept explicit zero for 0-based senders.
        raw_idx = int(digits)
        if raw_idx == 0:
            return 0
        if 1 <= raw_idx <= self.channel_count:
            return raw_idx - 1
        if 0 <= raw_idx < self.channel_count:
            return raw_idx
        return None

    def _push_to_engine(self):
        """Send values to the engine only when something changed."""
        if self._engine is None or not hasattr(self._engine, "set_channel_controls"):
            return

        with self._lock:
            faders = list(self.fader_values)
            dials = list(self.dial_values)

        if self._last_pushed_faders == faders and self._last_pushed_dials == dials:
            return

        self._engine.set_channel_controls(faders=faders, dials=dials)
        self._last_pushed_faders = list(faders)
        self._last_pushed_dials = list(dials)

        self._debug_print(f"push engine faders={faders[: self.channel_count]} dials={dials[: self.channel_count]}")


if __name__ == "__main__":
    bridge = SerialControlBridge()
    bridge.start()
    print("SerialControlBridge running (non-blocking). Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
