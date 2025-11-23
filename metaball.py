import pygame
import math
import random
import sys
import time
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

# optional fast PNG writer (Pillow). If missing, pygame.save is used instead.
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# -----------------------
# Configuration (tweak)
# -----------------------
WIDTH, HEIGHT = 1024, 1024       # window size
RENDER_W, RENDER_H = 512, 512    # internal render resolution (higher quality)
FPS_TARGET = 60
AUTOSAVE_INTERVAL_FRAMES = 10
METABALL_COUNT = 16
FIELD_THRESHOLD = 0.8
FRAMES_DIR = Path(r"C:\Users\aaadn\Downloads\Projects\GenAI\frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Pygame and thread setup
# -----------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Generative Art - Optimized")
clock = pygame.time.Clock()

save_queue = queue.Queue(maxsize=8)

def saver_thread_fn(q: queue.Queue):
    """Background thread: writes numpy RGB arrays to PNG files."""
    while True:
        item = q.get()
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
            print("Save error:", e)
        q.task_done()

saver = threading.Thread(target=saver_thread_fn, args=(save_queue,), daemon=True)
saver.start()

# -----------------------
# Metaballs storage
# -----------------------
class MetaBalls:
    """Holds arrays for metaball parameters and updates them each frame."""
    def __init__(self, n):
        self.n = n
        self.x = np.random.uniform(RENDER_W * 0.3, RENDER_W * 0.7, size=(n,)).astype(np.float32)
        self.y = np.random.uniform(RENDER_H * 0.3, RENDER_H * 0.7, size=(n,)).astype(np.float32)
        self.base_r = np.random.uniform(20, 50, size=(n,)).astype(np.float32)
        self.t_off_x = np.random.uniform(0, 100, size=(n,)).astype(np.float32)
        self.t_off_y = np.random.uniform(0, 100, size=(n,)).astype(np.float32)
        self.freq_x1 = np.random.uniform(0.01, 0.03, size=(n,)).astype(np.float32)
        self.freq_x2 = np.random.uniform(0.02, 0.05, size=(n,)).astype(np.float32)
        self.freq_y1 = np.random.uniform(0.01, 0.03, size=(n,)).astype(np.float32)
        self.freq_y2 = np.random.uniform(0.02, 0.05, size=(n,)).astype(np.float32)
        self.amp_x = np.random.uniform(RENDER_W * 0.18, RENDER_W * 0.42, size=(n,)).astype(np.float32)
        self.amp_y = np.random.uniform(RENDER_H * 0.18, RENDER_H * 0.42, size=(n,)).astype(np.float32)
        self.r = self.base_r.copy()

    def update(self, t):
        cx, cy = RENDER_W * 0.5, RENDER_H * 0.5
        self.x = cx + np.sin(t * self.freq_x1 + self.t_off_x) * (self.amp_x * 0.6) + \
                      np.cos(t * self.freq_x2) * (self.amp_x * 0.4)
        self.y = cy + np.cos(t * self.freq_y1 + self.t_off_y) * (self.amp_y * 0.6) + \
                      np.sin(t * self.freq_y2) * (self.amp_y * 0.4)
        self.r = self.base_r + np.sin(t * 0.05 + self.t_off_x) * 12.0

# -----------------------
# Renderer (vectorized)
# -----------------------
class ProFluidRenderer:
    """Computes field and maps to RGB using preallocated arrays."""
    def __init__(self, w, h, metaballs: MetaBalls, n):
        self.w = w
        self.h = h
        self.mballs = metaballs
        self.n = n

        xs = np.arange(self.w, dtype=np.float32)
        ys = np.arange(self.h, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing='xy')  # shape (h, w)
        self.X = np.ascontiguousarray(X.astype(np.float32))
        self.Y = np.ascontiguousarray(Y.astype(np.float32))

        self.field = np.zeros((self.h, self.w), dtype=np.float32)
        self.rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def compute_field(self):
        eps = 1e-2
        self.field.fill(0.0)
        xs = self.mballs.x[:, None, None]
        ys = self.mballs.y[:, None, None]
        rs = (self.mballs.r * self.mballs.r)[:, None, None]

        if NUMEXPR_AVAILABLE:
            for i in range(self.n):
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
            dist_sq = np.maximum(dist_sq, eps)
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

        self.rgb.fill(0)
        self.rgb[mask, 0] = np.clip((r + m) * 255.0, 0, 255).astype(np.uint8)
        self.rgb[mask, 1] = np.clip((g + m) * 255.0, 0, 255).astype(np.uint8)
        self.rgb[mask, 2] = np.clip((b + m) * 255.0, 0, 255).astype(np.uint8)

    def render_to_surface(self, surf, t):
        self.compute_field()
        self.field_to_rgb(t)
        arr_for_blit = np.swapaxes(self.rgb, 0, 1)  # to (w,h,3)
        pygame.surfarray.blit_array(surf, arr_for_blit)

# -----------------------
# Initialize
# -----------------------
mb = MetaBalls(METABALL_COUNT)
renderer = ProFluidRenderer(RENDER_W, RENDER_H, mb, METABALL_COUNT)
render_surface = pygame.Surface((RENDER_W, RENDER_H))

# -----------------------
# Main loop
# -----------------------
t = 0.0
frame = 0
running = True
gallery_mode = False
frames_saved = 0

mode_font = pygame.font.Font(None, 28)

while running:
    start = time.time()

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.key == pygame.K_g:
                gallery_mode = not gallery_mode
            elif ev.key == pygame.K_s and not gallery_mode:
                arr = renderer.rgb.copy()
                filename = FRAMES_DIR / f"manual_{frame:06d}.png"
                try:
                    save_queue.put_nowait((arr, str(filename)))
                except queue.Full:
                    pass

    screen.fill((0, 0, 0))

    if gallery_mode:
        saved_files = sorted(FRAMES_DIR.glob("*.png"))
        if saved_files:
            try:
                img = pygame.image.load(str(saved_files[-1])).convert()
                img = pygame.transform.scale(img, (WIDTH, HEIGHT))
                screen.blit(img, (0, 0))
            except Exception:
                pass
        else:
            font = pygame.font.Font(None, 48)
            txt = font.render("No frames yet", True, (255, 255, 255))
            screen.blit(txt, txt.get_rect(center=(WIDTH//2, HEIGHT//2)))
    else:
        mb.update(t)
        renderer.render_to_surface(render_surface, t)
        pygame.transform.scale(render_surface, (WIDTH, HEIGHT), screen)

        if frame % AUTOSAVE_INTERVAL_FRAMES == 0:
            arr = renderer.rgb.copy()
            filename = FRAMES_DIR / f"frame_{frame:06d}.png"
            try:
                save_queue.put_nowait((arr, str(filename)))
                frames_saved += 1
            except queue.Full:
                pass

    # show only the current mode (no FPS)
    mode_text = "GALLERY" if gallery_mode else "GENERATOR"
    mode_surf = mode_font.render(f"Mode: {mode_text}", True, (255, 255, 255))
    bg = pygame.Surface((mode_surf.get_width() + 12, mode_surf.get_height() + 8))
    bg.set_alpha(160); bg.fill((0, 0, 0))
    screen.blit(bg, (10, 10))
    screen.blit(mode_surf, (16, 14))

    pygame.display.flip()
    frame += 1
    t += 0.18
    clock.tick(FPS_TARGET)

# clean exit: tell saver to stop and wait shortly
save_queue.put(None)
saver.join(timeout=2)

pygame.quit()
sys.exit()
