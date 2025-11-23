# 🌌 Generative Metaball Art --- Raspberry Pi Optimized

A fully-optimized **generative art engine** designed specifically for
**Raspberry Pi 4/5**.\
This project renders real-time **metaball fluid visuals** using
NumPy-accelerated field calculations, a custom Pi-friendly render
pipeline, threaded autosaving, and an optional gallery slideshow/video
mode.

Built for **24/7 art installations**, **digital galleries**, **senior
design expos**, and **generative display walls**.

## ✨ Features

### 🎨 Real-Time Generative Art

-   Metaball fluid simulation\
-   Dynamic color-field shader (HSV → RGB)\
-   Smooth motion from sinusoidal field oscillations\
-   H/W-accelerated blitting via Pygame

### ⚡ Pi-Optimized Engine

-   384×384 internal render resolution (tunable)\
-   Optional **NumExpr acceleration** for per-ball field math\
-   Uses **preallocated arrays** to reduce garbage collection\
-   FPS target configurable (default 30)

### 💾 Threaded Autosaving

-   Saves frames every N frames (default: 5)\
-   Saves asynchronously using a background saver thread\
-   Backpressure prevents Pi overload\
-   PNG optimized via Pillow (if installed)

### 🖼 Gallery Mode (Slides + Video Renderer)

Press **G** at runtime to toggle: - Slideshow of all saved frames\
- Optional automatic MP4 compilation using OpenCV\
- Smooth playback at your target FPS

### 🧵 Clean System Integration

-   Graceful shutdown on SIGTERM/SIGINT\
-   Comes with a **systemd service** for auto-booting into the art
    display\
-   Perfect for 24/7 unattended installations

## 📂 Project Structure

    metaball_pi.py        # Main generative art engine
    frames/               # Auto-saved PNG frames (created at runtime)
    gallery_render.mp4    # Optional generated video
    genart.service        # (Included in script comments)

## 🛠 Requirements

### Python 3.9+

Required libraries:

    pygame
    numpy
    pillow
    numexpr
    opencv-python

## 🔧 Installation (Desktop or Pi)

Clone:

``` bash
git clone https://github.com/aaadnan259/Portfolio.git
cd Portfolio
```

Install:

``` bash
pip3 install pygame numpy pillow numexpr opencv-python
```

## 🚀 Running

``` bash
python3 metaball_pi.py
```

Fullscreen:

``` bash
python3 metaball_pi.py --fps 30
```

## 🎮 Controls

  Key          Action
  ------------ ---------------------
  **G**        Toggle gallery mode
  **ESC**      Quit
  **Ctrl+C**   Quit

## 🧾 Flags

  Flag                   Description
  ---------------------- -----------------------
  `--fps N`              FPS
  `--save-dir path`      Save directory
  `--no-autosave`        Disable autosave
  `--display-size WxH`   Output resolution
  `--gallery`            Start in gallery mode

## 🖥 Raspberry Pi Deployment

``` bash
mkdir -p /home/pi/generator
cp metaball_pi.py /home/pi/generator/metaball.py
```

Recommended `/boot/config.txt`:

    gpu_mem=256
    dtoverlay=vc4-fkms-v3d

Virtual environment:

``` bash
python3 -m venv /home/pi/generator/venv
/home/pi/generator/venv/bin/pip install --upgrade pip
/home/pi/generator/venv/bin/pip install pygame numpy pillow numexpr opencv-python
```

## 🔁 systemd Service

`/etc/systemd/system/genart.service`:

    [Unit]
    Description=Generative Art Display (metaball)
    After=graphical.target

    [Service]
    User=pi
    Environment=DISPLAY=:0
    Environment=XAUTHORITY=/home/pi/.Xauthority
    ExecStart=/usr/bin/python3 /home/pi/generator/metaball.py --fps 30
    Restart=always
    RestartSec=2

    [Install]
    WantedBy=graphical.target

Enable:

``` bash
sudo systemctl daemon-reload
sudo systemctl enable genart.service
sudo systemctl start genart.service
```

## 📜 License

MIT
