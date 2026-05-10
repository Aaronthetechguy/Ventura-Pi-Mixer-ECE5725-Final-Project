"""Python wrapper for the realtime C JACK audio engine.

This file starts audio_engine_rt2, sends packed EQ/gain/pan updates over a
Unix control socket, and polls meter data for the Pygame GUI.
"""

import os
import queue
import shutil
import socket
import struct
import subprocess
import threading
import time

C_BINARY = "./audio_engine_rt2"  # Set this to your compiled C executable if not in PATH
CONTROL_SOCKET_PATH = "/tmp/eq_control.sock"
METER_SOCKET_PATH = "/tmp/eq_meter.sock"

MAX_TRACKS = 4
NUM_BANDS = 3

CONTROL_STRUCT = struct.Struct(
    "4f" + "4f" + ("ii f f f f i" * (MAX_TRACKS * NUM_BANDS))
)
METER_STRUCT = struct.Struct("fffi" * (1 + MAX_TRACKS))
LEGACY_METER_STRUCT = struct.Struct("fffi")

FILTER_MAP = {
    "bell": 0,
    "peak": 0,
    "low_shelf": 1,
    "high_shelf": 2,
    "lp": 3,
    "lowpass": 3,
    "hp": 4,
    "highpass": 4,
}


class LiveAudioEngine:
    """Bridge between GUI state objects and the C realtime audio process."""

    def __init__(
        self,
        eq_states,
        input_devices=None,
        output_device=None,
        blocksize=64,
        fs=48000,
        c_binary=C_BINARY,
    ):
        # input_devices/output_device are intentionally ignored here because
        # the C runtime owns audio I/O (JACK ports/devices).
        self.fs = int(fs)
        self.blocksize = int(blocksize)
        self.c_binary = c_binary

        if not isinstance(eq_states, (list, tuple)):
            eq_states = [eq_states]
        else:
            eq_states = list(eq_states)

        self.num_tracks = max(1, min(MAX_TRACKS, len(eq_states)))
        self.eq_states = self._expand_eq_states(eq_states, self.num_tracks)

        self.active_track_index = 0
        self.command_q = queue.Queue()
        self.playing = True  # GUI expects this style to exist
        self._meter_cache = {
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "peak_hold_dbfs": -120.0,
            "clipped": False,
            "tracks": [
                {
                    "rms_dbfs": -120.0,
                    "peak_dbfs": -120.0,
                    "peak_hold_dbfs": -120.0,
                    "clipped": False,
                }
                for _ in range(MAX_TRACKS)
            ],
        }
        self._meter_lock = threading.Lock()
        self._stop = threading.Event()
        self._meter_thread = None
        self._proc = None
        self._resolved_c_binary = None

        self.track_labels = self._make_track_labels()
        self._control_lock = threading.Lock()
        self.track_faders = [1.0 for _ in range(MAX_TRACKS)]
        self.track_dials = [0.0 for _ in range(MAX_TRACKS)]
        self.track_muted = [False for _ in range(MAX_TRACKS)]
        self._saved_track_faders = [1.0 for _ in range(MAX_TRACKS)]
        self._last_sent_gains = None
        self.debug_controls = False

    def _expand_eq_states(self, eq_states, target_track_count):
        """Normalize caller-provided EQ states to the active track count."""
        if len(eq_states) == target_track_count:
            return eq_states

        if len(eq_states) == 1:
            return [eq_states[0] for _ in range(target_track_count)]

        if len(eq_states) > target_track_count:
            return eq_states[:target_track_count]

        raise ValueError(
            f"eq_states must have 1 or {target_track_count} entries; got {len(eq_states)}"
        )

    def _make_track_labels(self):
        """Build display labels for the active tracks."""
        return [f"Track {i + 1}" for i in range(self.num_tracks)]

    def get_track_labels(self):
        return list(self.track_labels)

    def get_active_track_index(self):
        return int(self.active_track_index)

    def set_active_track(self, index):
        index = max(0, min(int(index), len(self.eq_states) - 1))
        self.active_track_index = index
        return index

    def get_active_eq_state(self):
        return self.eq_states[self.active_track_index]

    def _wait_for_socket(self, path, timeout=5.0):
        """Wait until the C process creates a control or meter socket."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(path):
                return
            time.sleep(0.05)
        raise RuntimeError(f"Socket did not appear: {path}")

    def _resolve_c_binary(self):
        """Find the compiled C engine either by path or on PATH."""
        candidate = (self.c_binary or "").strip()
        if not candidate:
            raise ValueError("c_binary cannot be empty")

        if os.path.isabs(candidate) or os.sep in candidate or "/" in candidate:
            if os.path.isfile(candidate):
                return candidate
            raise FileNotFoundError(
                f"C binary not found at '{candidate}'. Build audio_engine_rt2.c to an executable and set c_binary to that path."
            )

        resolved = shutil.which(candidate)
        if resolved:
            return resolved

        raise FileNotFoundError(
            f"C binary '{candidate}' was not found in PATH. Build audio_engine_rt2.c and set c_binary to the executable path."
        )

    def _send_current_eq_to_c(self):
        """Pack current mixer controls and EQ bands into the C message ABI."""
        with self._control_lock:
            gains = [
                0.0 if self.track_muted[i] else float(self.track_faders[i])
                for i in range(self.num_tracks)
            ]
            dials = list(self.track_dials[: self.num_tracks])
            while len(gains) < MAX_TRACKS:
                gains.append(1.0)
            while len(dials) < MAX_TRACKS:
                dials.append(0.0)
        flat = list(gains) + list(dials)

        for ch in range(MAX_TRACKS):
            # The C struct is fixed-size, so inactive tracks still get defaults.
            if ch < len(self.eq_states):
                _, bands = self.eq_states[ch].get_snapshot()
            else:
                bands = []

            for b in range(NUM_BANDS):
                if b < len(bands):
                    band = bands[b]
                else:
                    band = {
                        "enabled": 0,
                        "type": "bell",
                        "freq": 1000.0,
                        "q": 1.0,
                        "gain_db": 0.0,
                        "slope": 1.0,
                        "order": 2,
                    }

                flat.extend([
                    int(band.get("enabled", 1)),
                    int(FILTER_MAP.get(band.get("type", "bell"), 0)),
                    float(band.get("freq", 1000.0)),
                    float(band.get("q", 1.0)),
                    float(band.get("gain_db", 0.0)),
                    float(band.get("slope", 1.0)),
                    int(band.get("order", 2)),
                ])

        payload = CONTROL_STRUCT.pack(*flat)

        # A short-lived socket keeps updates simple and avoids shared state.
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(CONTROL_SOCKET_PATH)
        sock.sendall(payload)
        sock.close()

        if self.debug_controls:
            print(f"sent control gains: {gains[: self.num_tracks]}, panning: {dials[: self.num_tracks]}")

    def set_channel_controls(self, faders=None, dials=None):
        """Receive hardware or GUI channel controls and trigger a C rebuild."""
        changed = False
        with self._control_lock:
            if faders is not None:
                for i, value in enumerate(list(faders)[: self.num_tracks]):
                    v = max(0.0, min(2.0, float(value)))
                    if self.track_faders[i] != v:
                        self.track_faders[i] = v
                        changed = True

            if dials is not None:
                for i, value in enumerate(list(dials)[: self.num_tracks]):
                    v = max(-1.0, min(1.0, float(value)))
                    if self.track_dials[i] != v:
                        self.track_dials[i] = v
                        changed = True

        if changed:
            if self.debug_controls:
                print(f"control update: faders={faders if faders is not None else None}, dials={dials if dials is not None else None}")
            self.request_rebuild()

    def get_channel_dial_values(self):
        """Get the current panning values for all channels."""
        with self._control_lock:
            return [float(self.track_dials[i]) for i in range(self.num_tracks)]

    def request_rebuild(self, track_index=None):
        # track_index is ignored because C rebuilds all active tracks together.
        self._send_current_eq_to_c()

    def set_track_muted(self, track_index, muted):
        """Mute/unmute by sending zero or saved fader gain to the C engine."""
        index = max(0, min(int(track_index), self.num_tracks - 1))
        muted = bool(muted)

        with self._control_lock:
            if muted and not self.track_muted[index]:
                self.track_muted[index] = True
            elif (not muted) and self.track_muted[index]:
                self.track_muted[index] = False
            else:
                self.track_muted[index] = muted

        self.request_rebuild()

    def set_track_mute(self, track_index, muted):
        self.set_track_muted(track_index, muted)

    def mute_track(self, track_index):
        self.set_track_muted(track_index, True)

    def unmute_track(self, track_index):
        self.set_track_muted(track_index, False)

    def get_output_meter(self):
        """Return the latest master meter values cached by the polling thread."""
        with self._meter_lock:
            return dict(self._meter_cache)

    def get_track_output_meter(self, track_index):
        """Return a single track meter, or silence if it is unavailable."""
        track_index = int(track_index)
        with self._meter_lock:
            tracks = self._meter_cache.get("tracks", [])
            if 0 <= track_index < min(self.num_tracks, len(tracks)):
                return dict(tracks[track_index])
        return {
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "peak_hold_dbfs": -120.0,
            "clipped": False,
        }

    def _recv_exact(self, sock, size):
        """Read exactly size bytes unless the socket closes early."""
        chunks = []
        received = 0
        while received < size:
            chunk = sock.recv(size - received)
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _meter_poll_loop(self):
        """Poll the C meter socket at GUI-friendly speed."""
        while not self._stop.is_set():
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(METER_SOCKET_PATH)
                sock.sendall(struct.pack("I", 1))
                data = self._recv_exact(sock, METER_STRUCT.size)
                sock.close()

                if len(data) == METER_STRUCT.size:
                    # New packet format: master meter followed by per-track meters.
                    values = METER_STRUCT.unpack(data)
                    master = values[0:4]
                    tracks = [values[i : i + 4] for i in range(4, len(values), 4)]
                    with self._meter_lock:
                        self._meter_cache = {
                            "rms_dbfs": master[0],
                            "peak_dbfs": master[1],
                            "peak_hold_dbfs": master[2],
                            "clipped": bool(master[3]),
                            "tracks": [
                                {
                                    "rms_dbfs": track[0],
                                    "peak_dbfs": track[1],
                                    "peak_hold_dbfs": track[2],
                                    "clipped": bool(track[3]),
                                }
                                for track in tracks[: self.num_tracks]
                            ],
                        }
                elif len(data) == LEGACY_METER_STRUCT.size:
                    # Old packet format: master meter only, keep track meters silent.
                    rms_dbfs, peak_dbfs, peak_hold_dbfs, clipped = LEGACY_METER_STRUCT.unpack(data)
                    with self._meter_lock:
                        self._meter_cache = {
                            "rms_dbfs": rms_dbfs,
                            "peak_dbfs": peak_dbfs,
                            "peak_hold_dbfs": peak_hold_dbfs,
                            "clipped": bool(clipped),
                            "tracks": [
                                {
                                    "rms_dbfs": -120.0,
                                    "peak_dbfs": -120.0,
                                    "peak_hold_dbfs": -120.0,
                                    "clipped": False,
                                }
                                for _ in range(self.num_tracks)
                            ],
                        }
            except Exception:
                pass

            time.sleep(0.03)  # ~33 Hz GUI meter


    def run(self):
        """Start the C process, initialize sockets, then service GUI commands."""
        # Resolve once so startup failures clearly indicate wrong executable path.
        self._resolved_c_binary = self._resolve_c_binary()

        # Your C file hardcodes app.num_tracks at startup, so this must match.
        # Change the C main if you want it to follow Python dynamically.
        self._proc = subprocess.Popen([self._resolved_c_binary, str(self.num_tracks)])

        # If process exits immediately, report that before waiting for sockets.
        time.sleep(0.1)
        rc = self._proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"C process exited early with code {rc}. Command: {self._resolved_c_binary} {self.num_tracks}"
            )

        self._wait_for_socket(CONTROL_SOCKET_PATH)
        self._wait_for_socket(METER_SOCKET_PATH)

        # Send initial EQ once
        self._send_current_eq_to_c()

        self._meter_thread = threading.Thread(target=self._meter_poll_loop, daemon=True)
        self._meter_thread.start()

        while not self._stop.is_set():
            try:
                cmd = self.command_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Your current C engine does not have play/pause/restart commands yet.
            # Keep them harmless so gui4.py still runs.
            if cmd == "restart":
                self._send_current_eq_to_c()
            elif cmd == "toggle_play":
                pass

    def close(self):
        """Stop background work and terminate the C engine process."""
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
