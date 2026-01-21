import pygame
import numpy as np
import cv2
import time
from typing import Tuple, Optional, List

class TrolleyPixelGUI:
    """
    Pixel art style GUI for the Trolley Problem using Pygame.
    """
    
    # Colors
    COLOR_BG = (50, 150, 50)  # Grass green
    COLOR_TRACK = (100, 100, 100) # Gray
    COLOR_TIE = (139, 69, 19) # Brown
    COLOR_TROLLEY = (200, 0, 0) # Red
    COLOR_PERSON_LEFT = (0, 165, 255) # Orange (matches original)
    COLOR_PERSON_RIGHT = (255, 100, 100) # Blue (matches original)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BUTTON = (200, 200, 200)
    COLOR_BUTTON_HOVER = (255, 255, 255)
    
    def __init__(self, window_width=1280, window_height=720):
        self.window_width = window_width
        self.window_height = window_height
        
        # Pygame init
        pygame.init()
        pygame.display.set_caption("Trolley Problem - Perspective View")
        self.screen = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)
        
        # Load Assets
        try:
            self.img_bg = pygame.image.load("assets/bg_perspective.png").convert()
            self.img_bg = pygame.transform.scale(self.img_bg, (window_width, window_height))
            
            self.img_trolley = pygame.image.load("assets/trolley_tram.png").convert_alpha()
            # Base size for the trolley (when scale is 1.0)
            self.trolley_base_w = 200
            self.trolley_base_h = 160
            self.img_trolley = pygame.transform.scale(self.img_trolley, (self.trolley_base_w, self.trolley_base_h))
            
            self.img_lever_left = pygame.image.load("assets/lever_left.png").convert_alpha()
            self.img_lever_left = pygame.transform.scale(self.img_lever_left, (150, 150))
            
            self.img_lever_right = pygame.image.load("assets/lever_right.png").convert_alpha()
            self.img_lever_right = pygame.transform.scale(self.img_lever_right, (150, 150))
            
            self.img_frame = pygame.image.load("assets/monitor_frame.png").convert_alpha()
            self.img_frame = pygame.transform.scale(self.img_frame, (360, 280))
            
            self.assets_loaded = True
        except Exception as e:
            print(f"Failed to load assets: {e}")
            self.assets_loaded = False
        
        # State placeholders
        self.trolley_pos = (0, 0) # x, y
        
    def _draw_trolley(self, state, decision, time_remaining):
        """Draw the trolley with perspective scaling."""
        if not self.assets_loaded:
            return

        # Animation Parameters
        # Path: Top-Left to Center Split
        # Based on new asset: Track starts top-left, splits near center (approx 640, 360) and goes bottom-right
        
        start_x, start_y = -100, -50 # Start off-screen top-left
        split_x, split_y = self.window_width // 2, self.window_height // 2
        
        # Scale: Far (0.2) to Near (1.0 at split point?) 
        # Actually it continues past split, so maybe 0.2 -> 0.6 at split -> 1.0 at end
        
        # Calculate progress (0.0 to 1.0) based on countdown
        if state == "COUNTDOWN":
            progress = max(0.0, min(1.0, (10.0 - time_remaining) / 10.0))
        elif state in ["DECIDING", "EXECUTING"]:
            progress = 1.0
        else: # IDLE
            progress = 0.0
            
        # Interpolate Position (Phase 1: Approaching Split)
        cur_x = start_x + (split_x - start_x) * progress
        cur_y = start_y + (split_y - start_y) * progress
        
        # Interpolate Scale
        cur_scale = 0.2 + (0.6 - 0.2) * progress
        
        # Execute phase: Divert logic (Phase 2: After Split)
        if state == "EXECUTING":
            # Continue movement past split
            # We need to animate PAST the split point based on time since execution started?
            # For now, let's just shift it further based on a simple increment if we had a timer.
            # Since we don't have an 'execution timer' passed in, we'll just show it at the end of the split.
            
            # Hack: Push it further along the chosen track
            # Let's assume we show the result state
            
            travel_dist = 200 # pixels past split
            
            if decision == "DIVERT_RIGHT":
                # Curve UP/RIGHT
                cur_x += travel_dist
                cur_y -= travel_dist * 0.5 
            else:
                # Continue STRAIGHT/DOWN-RIGHT matching diagonal
                cur_x += travel_dist
                cur_y += travel_dist
            
            cur_scale = 0.8 # Slightly bigger as it gets closer

        
        # Scale and Draw
        w = int(self.trolley_base_w * cur_scale)
        h = int(self.trolley_base_h * cur_scale)
        
        try:
            scaled_trolley = pygame.transform.scale(self.img_trolley, (w, h))
            # Center the trolley on the path point
            draw_x = cur_x - w // 2
            draw_y = cur_y - h // 2
            self.screen.blit(scaled_trolley, (int(draw_x), int(draw_y)))
        except Exception as e:
            print(f"Error drawing trolley: {e}")
            
    def render(self, frame_cv2: Optional[np.ndarray], 
               left_count: int, right_count: int, 
               left_conf: float, right_conf: float,
               time_remaining: float,
               state_name: str,
               decision: Optional[str] = None):
        """
        Main render loop.
        frame_cv2: BGR image from OpenCV
        """
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
        # Draw background
        if self.assets_loaded:
            self.screen.blit(self.img_bg, (0, 0))
            
            # Draw lever (bottom right or somewhere visible)
            # lever_pos = (self.window_width - 200, self.window_height - 200)
            # if decision == "DIVERT_RIGHT":
            #     self.screen.blit(self.img_lever_right, lever_pos)
            # else:
            #     self.screen.blit(self.img_lever_left, lever_pos)
        else:
            self.screen.fill(self.COLOR_BG)
            # self._draw_tracks() # Removed as tracks are in BG now

        self._draw_trolley(state_name, decision, time_remaining)
        
        # Draw camera feed (monitor style)
        monitor_w, monitor_h = 320, 240
        monitor_x = 40
        monitor_y = self.window_height - monitor_h - 40
        
        if frame_cv2 is not None:
            # Resize 
            frame_resized = cv2.resize(frame_cv2, (monitor_w, monitor_h))
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            # Create surface
            cam_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            self.screen.blit(cam_surf, (monitor_x, monitor_y))
        else:
            pygame.draw.rect(self.screen, (0,0,0), (monitor_x, monitor_y, monitor_w, monitor_h))
            
        # Draw frame overlay
        if self.assets_loaded:
            # Center the frame over the monitor rect
            frame_x = monitor_x - 20
            frame_y = monitor_y - 20
            self.screen.blit(self.img_frame, (frame_x, frame_y))
            
        # Draw text/UI (Outline for readability)
        def draw_text_with_outline(text, font, color, pos):
            text_surf = font.render(text, True, color)
            outline_surf = font.render(text, True, (0,0,0))
            x, y = pos
            for dx, dy in [(-2,-2), (-2,2), (2,-2), (2,2)]:
                self.screen.blit(outline_surf, (x+dx, y+dy))
            self.screen.blit(text_surf, (x, y))

        draw_text_with_outline(f"{time_remaining:.1f}s", self.big_font, self.COLOR_TEXT, (self.window_width//2 - 50, 50))
        draw_text_with_outline(f"State: {state_name}", self.font, self.COLOR_TEXT, (20, 20))
        
        # Draw people counts
        draw_text_with_outline(f"L: {left_count}", self.font, self.COLOR_PERSON_LEFT, (self.window_width//2 - 250, 200))
        draw_text_with_outline(f"R: {right_count}", self.font, self.COLOR_PERSON_RIGHT, (self.window_width//2 + 200, 200))
        
        if decision:
            d_color = (255, 255, 0)
            text_size = self.big_font.size(f"DECISION: {decision}")
            draw_text_with_outline(f"DECISION: {decision}", self.big_font, d_color, 
                                 (self.window_width//2 - text_size[0]//2, self.window_height//2))

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def quit(self):
        pygame.quit()
