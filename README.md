# OAK-D Eye & Hand Tracking Relay

A high-performance, close-to-real-time stereo vision tracking system utilizing the **OAK-D Pro (DepthAI) camera**. This application tracks 3D eye (iris) coordinates and monitors hand presence across four screen quadrants. It acts as a relay, streaming the processed tracking data via UDP to external applications.

## Key Features

* **Hybrid Eye Tracking (3D):** Uses *MediaPipe FaceMesh* for initial detection and **Lucas-Kanade Optical Flow** to track the relevant landmarks. This ensures rigid, high-speed tracking of the iris across frames without the overhead of running the neural network continuously. Points are triangulated into real 3D space.

* **Asynchronous Hand Quadrant Detection:** Uses *MediaPipe Hands* to detect hand presence in four screen zones (Top-Left, Bottom-Left, Top-Right, Bottom-Right). Runs entirely in a background thread to prevent main-loop blocking.

* **High FPS Architecture:** By only using the cpu heavy *MediaPipe* as a detector and switching to the fast *Optical Flow* for Tracking, the application maintains high and stable data rates of up to 65-100 results / second depending on the cpu used.

* **Real-time UDP Relay:** Broadcasts the triangulated 3D eye coordinates and the boolean hand quadrant states to configurable network destinations.

* **Visual Config UI:** Provides a live camera feed with tracking point visualization and sliders for configuring the camera and tracking settings.

## Prerequisites

* **Hardware:** Luxonis OAK-D Pro Stereo Camera
* **Python:** 3.8 or higher
* **Dependencies:**
    * `opencv-python` 
    * `numpy`
    * `mediapipe`
    * `depthai`
    * `orjson`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/appelr/oak-d_tracking_relay.git
   cd oakd-tracking-relay
   ```

2. Install the required Python packages:
   ```bash
   pip install -e .
   ```
## Configuration

The application is configured via the built in *Visual Config UI* or directly using the `config.json` file located in the root directory. 
Key parameters include:
* Camera ISO, Exposure and IR Laser
* CLAHE (Contrast Limited Adaptive Histogram Equalization) settings for IR enhancement.
* MediaPipe detection confidence thresholds

## Usage

Ensure your OAK-D camera is connected and run the main script:

```bash
start-tracking
```

### UI Controls
* The application launches a visual debug window.
* Hand presence is indicated by a colored, semi-transparent overlay in the respective screen quadrant.
* Press `Q` to cleanly exit the *Visual Config UI* and continue tracking in high speed.

## Network Output (UDP)

Ensure your receiving application listens to the port specified in `config.json`:

* **Eyes:** Transmitted as Triangulated 3D Points (`x, y, z`) based on the stereo baseline of the OAK-D.
* **Hands:** Transmitted as four boolean values representing the active quadrants (Top-Left, Bottom-Left, Top-Right, Bottom-Right).

---
*Developed for robust, real-time spatial interaction.*
