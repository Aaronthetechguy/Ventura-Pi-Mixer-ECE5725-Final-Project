"""Application entry point for the touchscreen live EQ mixer.

This file creates per-channel EQ state, starts the serial control bridge,
launches the realtime audio engine thread, and runs the Pygame GUI loop.
"""

import threading
import pygame

from eq_state import EQState
from audio_engine7_live import LiveAudioEngine
from gui6 import EQLiveGui
from IO_Parsing import SerialControlBridge


WIDTH = 800
HEIGHT = 480
INPUT_DEVICES = None
OUTPUT_DEVICE = None


def main():
    """Start audio, hardware controls, and the fullscreen GUI."""
    # Up to 2 USB devices x 2 mono channels each = up to 4 tracks.
    eq_states = [EQState() for _ in range(2)]
    # On Pi-class hardware, a larger blocksize is usually more stable (fewer overflows) but adds latency.
    audio_engine = LiveAudioEngine(eq_states=eq_states, blocksize=64)
    control_bridge = None

    try:
        control_bridge = SerialControlBridge(channel_count=2)
        control_bridge.debug = False
        control_bridge.bind_engine(audio_engine)
        control_bridge.start()
    except Exception as e:
        # Keep audio + GUI running even if serial controls are unavailable.
        print(f"Serial control disabled: {e}")

    audio_engine.debug_controls = False

    audio_thread = threading.Thread(target=audio_engine.run, daemon=True)
    audio_thread.start()

    # Fullscreen mode matches the Raspberry Pi touchscreen target.
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Live Dual-USB EQ")
    clock = pygame.time.Clock()

    gui = EQLiveGui(screen, eq_states[0], audio_engine)

    running = True
    while running:
        # clock.tick keeps drawing and event handling near 60 FPS.
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            gui.handle_event(event)

        gui.update(dt)
        gui.draw()
        pygame.display.flip()

    # Close hardware/audio resources before leaving pygame.
    if control_bridge is not None:
        control_bridge.close()
    audio_engine.close()
    pygame.quit()


if __name__ == "__main__":
    main()
