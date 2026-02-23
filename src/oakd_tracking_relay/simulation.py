import time
import math
from typing import Dict
from collections import deque

import matplotlib.pyplot as plt

from oakd_tracking_relay.ConfigurationManager import Configuration
from oakd_tracking_relay.UDPManager import UDP

def generate_smooth_point(t: float, speed_x: float, speed_y: float, speed_z: float, 
                          amp_x: float, amp_y: float, amp_z: float, 
                          offset_x: float, offset_y: float, offset_z: float) -> Dict[str, float]:
    """Generiert weiche, fortlaufende 3D-Koordinaten in Millimeter."""
    x = math.sin(t * speed_x) * amp_x + offset_x
    y = math.cos(t * speed_y) * amp_y + offset_y
    z = math.sin(t * speed_z) * amp_z + offset_z
    return {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}

def main():
    config = Configuration.load("config.json")
    udpManager = UDP(config)

    # --- PLOT SETUP ---
    plt.ion() # Interaktiver Modus an (blockiert das Skript nicht)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Live Auge-Tracking (X/Y Radar)")
    ax.set_xlabel("X (Millimeter)")
    ax.set_ylabel("Y (Millimeter)")
    
    # Achsen-Limits festlegen (passend zu unseren Amplituden)
    ax.set_xlim(-350, 350)
    ax.set_ylim(-250, 250)
    ax.grid(True)

    # Wir merken uns die letzten 100 Punkte für einen "Schweif"-Effekt
    history_length = 100
    hist_x_L, hist_y_L = deque(maxlen=history_length), deque(maxlen=history_length)
    hist_x_R, hist_y_R = deque(maxlen=history_length), deque(maxlen=history_length)

    # Zwei Linien-Objekte für linkes (blau) und rechtes (rot) Auge anlegen
    lineL, = ax.plot([], [], 'b-', marker='o', markersize=4, label='Iris Links')
    lineR, = ax.plot([], [], 'r-', marker='o', markersize=4, label='Iris Rechts')
    ax.legend(loc="upper right")

    print("Starte geschmeidige Koordinaten-Simulation inkl. Plot... (Abbruch mit STRG+C oder Fenster schließen)")
    start_time = time.monotonic()

    try:
        # Läuft solange, bis jemand das Plot-Fenster schließt
        while plt.fignum_exists(fig.number):
            t = time.time()
            timestamp_ms = t * 1000.0

            # --- KOORDINATEN BERECHNEN ---
            head_center = generate_smooth_point(t, speed_x=0.8, speed_y=0.5, speed_z=0.3,
                                                amp_x=250.0, amp_y=150.0, amp_z=200.0,
                                                offset_x=0.0, offset_y=0.0, offset_z=600.0)
            
            irisL = {"x": round(head_center["x"] - 32.5, 2), "y": head_center["y"], "z": head_center["z"]}
            irisR = {"x": round(head_center["x"] + 32.5, 2), "y": head_center["y"], "z": head_center["z"]}

            handL = generate_smooth_point(t, speed_x=1.2, speed_y=1.5, speed_z=0.9,
                                          amp_x=200.0, amp_y=200.0, amp_z=100.0,
                                          offset_x=-150.0, offset_y=100.0, offset_z=450.0)

            handR = generate_smooth_point(t, speed_x=1.1, speed_y=1.6, speed_z=0.7,
                                          amp_x=200.0, amp_y=200.0, amp_z=100.0,
                                          offset_x=150.0, offset_y=100.0, offset_z=450.0)

            # --- UDP SENDEN ---
            udpManager.send(irisL, irisR, handL, handR, timestamp_ms)

            # --- PLOT UPDATEN ---
            # 1. Neue Werte an die Historie anhängen
            hist_x_L.append(irisL["x"])
            hist_y_L.append(irisL["y"])
            hist_x_R.append(irisR["x"])
            hist_y_R.append(irisR["y"])

            # 2. Die Daten im Plot aktualisieren
            lineL.set_data(hist_x_L, hist_y_L)
            lineR.set_data(hist_x_R, hist_y_R)

            # 3. Fenster neu zeichnen und kurz pausieren (ersetzt time.sleep)
            # 0.033 Sekunden entsprechen ~30 FPS
            plt.pause(0.033)

    except KeyboardInterrupt:
        print("\nSimulation abgebrochen.")
    finally: 
        udpManager.close()
        plt.close('all')

if __name__ == "__main__":
    main()