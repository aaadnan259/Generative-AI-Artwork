#!/usr/bin/env python3
"""
Pi-optimized metaball generator
- Replaces original metaball.py for Raspberry Pi 4 installations (headless/gallery)

Features:
- Reduced allocations: reuses arrays and surfaces
- Lower default render resolution and FPS for Pi stability
- Uses numexpr if available for per-ball math speedup
- Threaded PNG saver with backpressure
- Graceful SIGTERM/SIGINT handling for systemd
- Fullscreen, borderless, no-cursor gallery mode
- CLI flags: --gallery, --save-dir, --no-autosave, --fps

ALSO INCLUDED (below this script in the same document):
- systemd service file content (genart.service)
- recommended /boot/config.txt tweaks
- commands to install dependencies and deploy to /home/pi/generator/

Save this file as /home/pi/generator/metaball.py on your Pi (or replace your existing file).
"""

import os
import sys
import math
import time
import signal
import argparse
import threading
import queue
from pathlib import Path

import numpy as np

# try to use numexpr for faster per-ball math; fall back to numpy if not available
try:
    import numexpr as ne
    NUMEXPR_AVAILABLE = True
except Exception:
    NUMEXPR_AVAILABLE = False

# optional fast PNG writer (Pillow). If missing, fall back to pygame's saving.
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Try importing OpenCV for video compilation
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not found. Video compilation will be disabled.")

# Delayed import of pygame to allow headless usages or faster startup checks
try:
    import pygame
except Exception as e:
    raise SystemExit("pygame is required. Install with: pip3 install pygame")

# -----------------------
# Config (tune for Pi)
# -----------------------
# Keep internal render resolution modest for Pi 4 — you can increase slightly if you have a Pi 4 8GB
RENDER_W, RENDER_H = 384, 384
# The display scaling target (set to your LCD native resolution if you want)
DISPLAY_W, DISPLAY_H = 1024, 1024
FPS_TARGET = 30
AUTOSAVE_INTERVAL_FRAMES = 5  # saves every 5 frames
METABALL_COUNT = 12
FIELD_THRESHOLD = 0.8

# Where to store frames (default: in user's home generator/frames)
DEFAULT_FRAMES_DIR = Path.home() / "generator" / "frames"
DEFAULT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Threaded saver
# -----------------------
save_queue = queue.Queue(maxsize=4)
stop_saver = threading.Event()


def saver_thread_fn(q: queue.Queue, stop_event: threading.Event):
    """Background thread: writes numpy RGB arrays to PNG files."""
    while not stop_event.is_set():
        try:
            item = q.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            break
        arr, filename = item
        try:
            if PIL_AVAILABLE:
                Image.fromarray(arr).save(filename, "PNG", optimize=True)
            else:
                surf = pygame.surfarray.make_surface(np.swapaxes(arr, 0, 1))
                pygame.image.save(surf, filename)
        except Exception as e:
            print("[saver] Save error:", e)
        finally:
            q.task_done()

saver = threading.Thread(target=saver_thread_fn, args=(save_queue, stop_saver), daemon=True)
# don't start until main initializes pygame

# -----------------------
# Metaballs storage
# -----------------------
class MetaBalls:
    def __init__(self, n, w, h):
        self.n = n
        self.w = w
        self.h = h
        rng = np.random.default_rng(seed=None)
        self.x = rng.uniform(w * 0.3, w * 0.7, size=(n,)).astype(np.float32)
        self.y = rng.uniform(h * 0.3, h * 0.7, size=(n,)).astype(np.float32)
        self.base_r = rng.uniform(14, 36, size=(n,)).astype(np.float32)
        self.t_off_x = rng.uniform(0, 100, size=(n,)).astype(np.float32)
        self.t_off_y = rng.uniform(0, 100, size=(n,)).astype(np.float32)
        self.freq_x1 = rng.uniform(0.01, 0.03, size=(n,)).astype(np.float32)
        self.freq_x2 = rng.uniform(0.02, 0.05, size=(n,)).astype(np.float32)
        self.freq_y1 = rng.uniform(0.01, 0.03, size=(n,)).astype(np.float32)
        self.freq_y2 = rng.uniform(0.02, 0.05, size=(n,)).astype(np.float32)
        self.amp_x = rng.uniform(w * 0.16, w * 0.36, size=(n,)).astype(np.float32)
        self.amp_y = rng.uniform(h * 0.16, h * 0.36, size=(n,)).astype(np.float32)
        self.r = self.base_r.copy()

    def update(self, t):
        cx, cy = self.w * 0.5, self.h * 0.5
        # vectorized motion update
        self.x = cx + np.sin(t * self.freq_x1 + self.t_off_x) * (self.amp_x * 0.6) + \
                 np.cos(t * self.freq_x2) * (self.amp_x * 0.4)
        self.y = cy + np.cos(t * self.freq_y1 + self.t_off_y) * (self.amp_y * 0.6) + \
                 np.sin(t * self.freq_y2) * (self.amp_y * 0.4)
        self.r = self.base_r + np.sin(t * 0.05 + self.t_off_x) * 10.0

# -----------------------
# Renderer
# -----------------------
class PiRenderer:
    def __init__(self, w, h, metaballs: MetaBalls):
        self.w = w
        self.h = h
        self.m = metaballs

        xs = np.arange(self.w, dtype=np.float32)
        ys = np.arange(self.h, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing='xy')  # shape (h, w)
        # make contiguous arrays used by numpy/ne
        self.X = np.ascontiguousarray(X.astype(np.float32))
        self.Y = np.ascontiguousarray(Y.astype(np.float32))

        # preallocated buffers
        self.field = np.zeros((self.h, self.w), dtype=np.float32)
        self.rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def compute_field(self):
        eps = 1e-2
        # reset
        self.field.fill(0.0)

        xs = self.m.x[:, None, None]
        ys = self.m.y[:, None, None]
        rs = (self.m.r * self.m.r)[:, None, None]

        if NUMEXPR_AVAILABLE:
            # numexpr gives big speedups for heavy per-element math
            for i in range(self.m.n):
                expr = "r / ((X - x)**2 + (Y - y)**2 + eps)"
                local = {
                    'r': float(rs[i,0,0]),
                    'X': self.X,
                    'Y': self.Y,
                    'x': float(xs[i,0,0]),
                    'y': float(ys[i,0,0]),
                    'eps': eps
                }
                self.field += ne.evaluate(expr, local_dict=local)
        else:
            dx = self.X[None, :, :] - xs
            dy = self.Y[None, :, :] - ys
            dist_sq = dx * dx + dy * dy
            # clamp to avoid overflow and cheap branch
            dist_sq = np.maximum(dist_sq, eps)
            # sum of r / dist_sq across balls
            self.field = np.sum(rs / dist_sq, axis=0)

    def field_to_rgb(self, t):
        mask = (self.field >= FIELD_THRESHOLD)
        if not np.any(mask):
            self.rgb.fill(0)
            return

        intensity = np.clip(self.field[mask] / 4.0, 0.0, 1.0)
        x_coords = self.X[mask]
        y_coords = self.Y[mask]

        hue = (t * 2.0 + intensity * 120.0 + x_coords * 0.2 + y_coords * 0.2) % 360.0
        h = hue / 60.0
        s = 0.8 + intensity * 0.2
        v = 0.6 + intensity * 0.4
        c = v * s
        x_val = c * (1.0 - np.abs(h % 2 - 1.0))
        m = v - c

        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)

        cond1 = h < 1
        cond2 = (h >= 1) & (h < 2)
        cond3 = (h >= 2) & (h < 3)
        cond4 = (h >= 3) & (h < 4)
        cond5 = (h >= 4) & (h < 5)
        cond6 = h >= 5

        r[cond1], g[cond1], b[cond1] = c[cond1], x_val[cond1], 0
        r[cond2], g[cond2], b[cond2] = x_val[cond2], c[cond2], 0
        r[cond3], g[cond3], b[cond3] = 0, c[cond3], x_val[cond3]
        r[cond4], g[cond4], b[cond4] = 0, x_val[cond4], c[cond4]
        r[cond5], g[cond5], b[cond5] = x_val[cond5], 0, c[cond5]
        r[cond6], g[cond6], b[cond6] = c[cond6], 0, x_val[cond6]

        # write into preallocated rgb buffer
        self.rgb.fill(0)
        self.rgb[mask, 0] = np.clip((r + m) * 255.0, 0, 255).astype(np.uint8)
        self.rgb[mask, 1] = np.clip((g + m) * 255.0, 0, 255).astype(np.uint8)
        self.rgb[mask, 2] = np.clip((b + m) * 255.0, 0, 255).astype(np.uint8)

    def render_to_surface(self, surf, t):
        self.compute_field()
        self.field_to_rgb(t)
        # arr_for_blit must be shape (w, h, 3) for blit_array
        arr_for_blit = np.swapaxes(self.rgb, 0, 1)
        # blit directly into surface
        pygame.surfarray.blit_array(surf, arr_for_blit)

# -----------------------
# Utility: graceful shutdown
# -----------------------
shutdown_event = threading.Event()


def _signal_handler(signum, frame):
    print(f"Signal {signum} received, shutting down...")
    shutdown_event.set()

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# -----------------------
# Video Compilation
# -----------------------
def compile_video(frames_dir, output_path, fps=30):
    if not OPENCV_AVAILABLE:
        print("Cannot compile video: OpenCV not installed.")
        return False

    images = sorted(list(frames_dir.glob("frame_*.png")))
    if not images:
        print("No frames to compile.")
        return False

    print(f"Compiling {len(images)} frames into {output_path}...")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(str(images[0]))
    height, width, layers = first_frame.shape
    size = (width, height)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(str(output_path), fourcc, fps, size)

    for img_path in images:
        frame = cv2.imread(str(img_path))
        out.write(frame)

    out.release()
    print("Video compilation complete.")
    return True

# -----------------------
# Main application
# -----------------------

def main(argv):
    global AUTOSAVE_INTERVAL_FRAMES, FPS_TARGET

    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery', action='store_true', help='Start in gallery mode (show last saved image)')
    parser.add_argument('--save-dir', default=str(DEFAULT_FRAMES_DIR), help='Directory to save frames')
    parser.add_argument('--no-autosave', action='store_true', help='Disable autosaving frames')
    parser.add_argument('--fps', type=int, default=FPS_TARGET, help='FPS target')
    parser.add_argument('--display-size', type=str, default=f"{DISPLAY_W}x{DISPLAY_H}", help='WxH display size (for scaling)')
    args = parser.parse_args(argv[1:])

    frames_dir = Path(args.save_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    FPS_TARGET = max(10, min(60, args.fps))
    
    # Start pygame and configure display
    if sys.platform.startswith('linux'):
        os.environ.setdefault('SDL_AUDIODRIVER', 'dsp')  # avoid audio init delays on some systems
    
    print("Initializing pygame...")
    pygame.init()
    pygame.mouse.set_visible(False)

    # Fullscreen borderless window
    display_w, display_h = map(int, args.display_size.split('x'))
    flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
    try:
        screen = pygame.display.set_mode((display_w, display_h), flags)
    except Exception:
        # fallback to windowed if fullscreen fails
        print("Fullscreen failed, falling back to windowed mode")
        screen = pygame.display.set_mode((display_w, display_h))

    pygame.display.set_caption("Generative Art - Pi Optimized")

    # Create render surface (same pixel format as screen for blit_array)
    render_surface = pygame.Surface((RENDER_W, RENDER_H))

    mb = MetaBalls(METABALL_COUNT, RENDER_W, RENDER_H)
    renderer = PiRenderer(RENDER_W, RENDER_H, mb)

    # create pygame surface backed by same-sized array: faster blits
    # start saver thread now that pygame is initialized
    if not saver.is_alive():
        saver.start()

    clock = pygame.time.Clock()
    frame_count = 0
    t = 0.0
    
    gallery_mode = args.gallery
    slideshow_index = 0
    slideshow_timer = 0
    slideshow_images = []

    # If starting in gallery mode, load images immediately
    if gallery_mode:
        slideshow_images = sorted(list(frames_dir.glob("frame_*.png")))

    last_save_time = time.time()

    print("Entering main loop...")
    while not shutdown_event.is_set():
        start = time.time()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                shutdown_event.set()
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    shutdown_event.set()
                elif ev.key == pygame.K_g:
                    # Toggle gallery mode
                    gallery_mode = not gallery_mode
                    if gallery_mode:
                        # Enter gallery mode: Compile video and load images
                        print("Entering gallery mode...")
                        
                        # Compile video
                        video_path = frames_dir / "gallery_render.mp4"
                        compile_video(frames_dir, video_path, fps=FPS_TARGET)
                        
                        # Load images for slideshow
                        slideshow_images = sorted(list(frames_dir.glob("frame_*.png")))
                        slideshow_index = 0
                        if not slideshow_images:
                            print("No frames found for slideshow.")
                    else:
                        print("Exiting gallery mode...")

        if gallery_mode:
            # Slideshow logic
            if slideshow_images:
                # Change slide every 5 frames (approx 6 times a second at 30fps) for a fast slideshow,
                # or maybe slower? The user said "smooth as the render", so maybe play at FPS_TARGET.
                # Let's try to play at FPS_TARGET.
                
                try:
                    img_path = slideshow_images[slideshow_index]
                    img = pygame.image.load(str(img_path)).convert()
                    img = pygame.transform.scale(img, (display_w, display_h))
                    screen.blit(img, (0, 0))
                except Exception as e:
                    print(f"Error showing slide: {e}")

                slideshow_index = (slideshow_index + 1) % len(slideshow_images)
            else:
                 # No images to show
                screen.fill((0,0,0))
                font = pygame.font.Font(None, 36)
                text = font.render("No frames to display", True, (255, 255, 255))
                text_rect = text.get_rect(center=(display_w/2, display_h/2))
                screen.blit(text, text_rect)

            # Overlay mode text
            mode_font = pygame.font.Font(None, 24)
            mode_surf = mode_font.render("GALLERY MODE", True, (255, 255, 0))
            screen.blit(mode_surf, (8, 8))

            pygame.display.flip()
            clock.tick(FPS_TARGET) # Play at target FPS

        else:
            # Generator logic
            mb.update(t)
            renderer.render_to_surface(render_surface, t)

            # scale to display once per frame (smoothscale can be slower but looks better)
            try:
                scaled = pygame.transform.smoothscale(render_surface, (display_w, display_h))
            except Exception:
                scaled = pygame.transform.scale(render_surface, (display_w, display_h))

            screen.blit(scaled, (0, 0))

            # overlay mode text (small)
            mode_font = pygame.font.Font(None, 24)
            mode_surf = mode_font.render("GENERATOR", True, (255, 255, 255))
            screen.blit(mode_surf, (8, 8))

            pygame.display.flip()

            # autosave
            if (not args.no_autosave) and (frame_count % AUTOSAVE_INTERVAL_FRAMES == 0):
                arr = renderer.rgb.copy()
                filename = frames_dir / f"frame_{frame_count:08d}.png"
                try:
                    save_queue.put_nowait((arr, str(filename)))
                except queue.Full:
                    # drop frame if saver busy
                    pass

            frame_count += 1
            t += 0.12

            # maintain FPS target using clock.tick which sleeps appropriately
            clock.tick(FPS_TARGET)

    cleanup_and_exit()


def cleanup_and_exit():
    print("Shutting down: flushing save queue and quitting pygame...")
    # tell saver to stop
    stop_saver.set()
    try:
        # put sentinel and wait shortly
        save_queue.put_nowait(None)
    except Exception:
        pass
    if saver.is_alive():
        saver.join(timeout=2.0)

    try:
        pygame.quit()
    except Exception:
        pass
    sys.exit(0)


if __name__ == '__main__':
    main(sys.argv)


# -----------------------
# systemd service file (copy this content into /etc/systemd/system/genart.service)
# -----------------------
# [Unit]
# Description=Generative Art Display (metaball)
# After=graphical.target
#
# [Service]
# User=pi
# Environment=DISPLAY=:0
# Environment=XAUTHORITY=/home/pi/.Xauthority
# ExecStart=/usr/bin/python3 /home/pi/generator/metaball.py --fps 30
# Restart=always
# RestartSec=2
#
# [Install]
# WantedBy=graphical.target

# -----------------------
# Recommended /boot/config.txt tweaks (add these lines)
# -----------------------
# gpu_mem=256
# dtoverlay=vc4-fkms-v3d
# # Optional modest overclock for Pi4 (use at your own risk)
# # over_voltage=2
# # arm_freq=1850

# -----------------------
# Deployment commands (run on your Pi shell)
# -----------------------
# mkdir -p /home/pi/generator
# cp metaball.py /home/pi/generator/metaball.py
# sudo chown -R pi:pi /home/pi/generator
# python3 -m venv /home/pi/generator/venv
# /home/pi/generator/venv/bin/pip install --upgrade pip
# /home/pi/generator/venv/bin/pip install pygame numpy pillow numexpr opencv-python
# sudo cp genart.service /etc/systemd/system/genart.service   # create file from above content
# sudo systemctl daemon-reload
# sudo systemctl enable genart.service
# sudo systemctl start genart.service

# End of document
