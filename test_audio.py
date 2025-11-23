import os
import pygame

os.environ['SDL_AUDIODRIVER'] = 'dsp'
try:
    pygame.init()
    print("pygame init success")
except Exception as e:
    print(f"pygame init failed: {e}")
