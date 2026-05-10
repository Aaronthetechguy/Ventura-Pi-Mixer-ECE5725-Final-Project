#!/usr/bin/env bash
# Launch script for the Raspberry Pi live EQ mixer.
#
# It starts JACK with the selected USB audio interface, activates the Python
# virtual environment, runs the GUI/audio program, connects JACK ports, and
# shuts the Pi down when the session exits.

set -e

# Project and virtual environment locations on the Raspberry Pi.
PROJECT_DIR="/home/pi/EQ_test/Pi_EQ"
VENV_DIR="$PROJECT_DIR/venv"

# JACK settings: edit JACK_DEVICE if ALSA reports a different card number.
JACK_DEVICE="hw:1"
JACK_RATE="48000"
JACK_PERIOD="128"
JACK_NPERIODS="2"

# Python program and JACK client name created by audio_engine_rt2.c.
PYTHON_SCRIPT="main5_live.py"
CLIENT_NAME="usb_eq_mixer"

cleanup() {
    # Stop the Python app first, then stop JACK before powering down.
    echo "Cleaning up..."

    if [[ -n "$PY_PID" ]] && kill -0 "$PY_PID" 2>/dev/null; then
        kill "$PY_PID" 2>/dev/null || true
    fi

    killall jackd 2>/dev/null || true

    sudo shutdown 0 # Turn off power.
}

trap cleanup EXIT INT TERM

echo "Starting JACK..."
# Run JACK in realtime mode so the C callback gets stable low-latency timing.
jackd -R -P70 -dalsa -d"$JACK_DEVICE" -r"$JACK_RATE" -p"$JACK_PERIOD" -n"$JACK_NPERIODS" &
JACK_PID=$!

echo "Waiting for JACK server..."
until jack_lsp >/dev/null 2>&1; do # Discard output while waiting.
    sleep 0.2
done

echo "JACK is running."

cd "$PROJECT_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Starting Python EQ..."
# Python starts the C JACK client and the fullscreen GUI.
python3 "$PYTHON_SCRIPT" &
PY_PID=$!

echo "Waiting for JACK client: $CLIENT_NAME..."
# Do not connect ports until all expected JACK ports exist.
until jack_lsp | grep -qx "$CLIENT_NAME:in1" \
   && jack_lsp | grep -qx "$CLIENT_NAME:in2" \
   && jack_lsp | grep -qx "$CLIENT_NAME:outL" \
   && jack_lsp | grep -qx "$CLIENT_NAME:outR"; do

    if ! kill -0 "$PY_PID" 2>/dev/null; then
        echo "Python program exited before all JACK ports appeared."
        exit 1
    fi

    sleep 0.2
done

echo "Connecting JACK ports..."

# Wire two mono captures into the EQ and return stereo output to playback.
jack_connect system:capture_1 "$CLIENT_NAME:in1" || true
jack_connect system:capture_2 "$CLIENT_NAME:in2" || true
jack_connect "$CLIENT_NAME:outL" system:playback_1 || true
jack_connect "$CLIENT_NAME:outR" system:playback_2 || true

echo "Connected. Press Ctrl+C to stop."

# Keep this script alive until the GUI/audio process exits.
wait "$PY_PID"
