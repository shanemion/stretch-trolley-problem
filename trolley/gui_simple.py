import pygame
import cv2
import numpy as np
import time
from typing import Optional, Tuple

class TrolleySimpleGUI:
    """
    New 'Simple Theme' GUI for Trolley Problem.
    Perspective: Center, tracks splitting L/R. Left track straight.
    Assets: Clean minimal line art - artistic presentation.
    """
    
    # Colors - Clean Artistic Theme
    COLOR_BG = (255, 255, 255) # White background
    COLOR_TEXT = (30, 30, 30) # Near black text
    COLOR_FRAME = (50, 50, 50) # Dark gray frame
        
    def __init__(self, window_width=1280, window_height=720):
        self.window_width = window_width
        self.window_height = window_height
        
        # Pygame init
        pygame.init()
        pygame.display.set_caption("Trolley Problem - Simple Diagram View")
        self.screen = pygame.display.set_mode((window_width, window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        self.big_font = pygame.font.Font(None, 96)
        
        # Load Assets
        try:
            self.img_bg = pygame.image.load("assets/tracks_front_perspective.png").convert()
            # Scale background to fit vertically, center horizontally
            bg_w = self.img_bg.get_width()
            bg_h = self.img_bg.get_height()
            scale_bg = window_height / bg_h
            new_w = int(bg_w * scale_bg)
            self.img_bg = pygame.transform.scale(self.img_bg, (new_w, window_height))
            self.bg_x = (window_width - new_w) // 2
            
            self.img_trolley = pygame.image.load("assets/simple_trolley.png").convert_alpha()
            self.trolley_base_w = self.img_trolley.get_width()
            self.trolley_base_h = self.img_trolley.get_height()
            
            self.assets_loaded = True
        except Exception as e:
            print(f"Failed to load simple assets: {e}")
            self.assets_loaded = False
        
        self.start_time = None
        
    def _draw_trolley(self, state, decision, time_remaining):
        if not self.assets_loaded:
            return

        # Animation State
        # Path: Start Top-Center (small) -> Move Down -> Split L/R
        
        center_x = self.window_width // 2
        
        # Y-Positions
        start_y = 50
        # Split point is now lower (approx 40% down the screen based on new asset)
        split_y = int(self.window_height * 0.45) 
        end_y = self.window_height - 100
        
        # Scales
        start_scale = 0.1
        split_scale = 0.5
        end_scale = 1.0
        
        # Calculate Progress
        if state == "COUNTDOWN":
            # 0.0 to 1.0 over 10s
            progress = max(0.0, min(1.0, (10.0 - time_remaining) / 10.0))
        elif state in ["DECIDING", "EXECUTING"]:
            progress = 1.0 # Past the countdown, start splitting
        else:
            progress = 0.0
            
        # Draw Phase 1: Approaching Split (Straight Down)
        # We want the trolley to reach the split point roughly when countdown ends?
        # Or maybe it moves continuously. Let's say countdown brings it to the split point.
        
        # Interpolate Y
        cur_y = start_y + (split_y - start_y) * progress
        cur_scale = start_scale + (split_scale - start_scale) * progress
        cur_x = center_x
        
        # Draw Phase 2: Execution (Past Split)
        if state == "EXECUTING":
            # Animate past split
            # We don't have a precise execution timer passed in, so we simulate movement
            # In a real game loop we'd track state_time.
            # Hack: Just push it to end for visual feedback
            
            # TODO: Add execution timer to render args if we want smooth animation here
            # For now, put it further down the track
            
            cur_y = split_y + 150 # Moved past split (further down for visual confirmation)
            cur_scale = 0.7
            
            # Straight Left vs Curved Right
            if decision == "DIVERT_RIGHT":
                # Curve Right
                offset_x = 200 # Move right
                cur_x += offset_x
            else:
                # Straight Left (Diagonal)
                # Left track goes from Center Top to Bottom Left
                # Start X = Center, End X = Left Border?
                # Actually, the track image dictates the path.
                # Left track is a straight diagonal line.
                # So we simply interpolate X based on Y progress further.
                # Current logic was centered until split.
                # Let's say loop moves further along that vector.
                
                # Assume straight line equation continuation
                # Slope approx: (BottomLeftX - TopCenterX) / Height
                # Let's approximate visually based on scale
                cur_x -= 150 # Move left along the diagonal
        
        # Draw Trolley
        w = int(self.trolley_base_w * cur_scale)
        h = int(self.trolley_base_h * cur_scale)
        
        try:
            scaled = pygame.transform.smoothscale(self.img_trolley, (w, h))
            draw_pos = (int(cur_x - w//2), int(cur_y - h//2))
            self.screen.blit(scaled, draw_pos)
        except:
            pass
            

    def _draw_text_centered(self, text, font, color, y_pos):
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(self.window_width // 2, y_pos))
        self.screen.blit(surf, rect)

    def render(self, frame_cv2: Optional[np.ndarray], 
               left_count: int, right_count: int, 
               state_name: str, time_remaining: float,
               decision: str = "DEFAULT"):
        
        # Handling quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
        
        # Draw White BG
        self.screen.fill(self.COLOR_BG)
        
        # Draw Tracks (Centered)
        if self.assets_loaded:
            self.screen.blit(self.img_bg, (self.bg_x, 0))
            
        # Draw Camera Feed (Top Left corner, simple border)
        if frame_cv2 is not None:
             # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
            frame_rgb = np.rot90(frame_rgb)
            
            # Create Surface
            frame_surface = pygame.surfarray.make_surface(frame_rgb)
            frame_surface = pygame.transform.scale(frame_surface, (320, 240))
            
            # Draw border
            border_rect = pygame.Rect(10, 10, 320 + 4, 240 + 4)
            pygame.draw.rect(self.screen, self.COLOR_FRAME, border_rect)
            self.screen.blit(frame_surface, (12, 12))
            
        # Draw Overlay Info
        # Time
        if state_name == "COUNTDOWN":
            self._draw_text_centered(f"{time_remaining:.1f}s", self.big_font, self.COLOR_TEXT, 100)
            
        # State
        state_surf = self.font.render(f"State: {state_name}", True, self.COLOR_TEXT)
        self.screen.blit(state_surf, (self.window_width - 300, 20))
        
        # Trolley
        self._draw_trolley(state_name, decision, time_remaining)
        
        # Counts (Simple layout: Left count on left, Right count on right)
        left_text = self.font.render(str(left_count), True, self.COLOR_TEXT)
        right_text = self.font.render(str(right_count), True, self.COLOR_TEXT)
        
        # Position near the bottom tracks
        self.screen.blit(left_text, (self.window_width // 2 - 200, self.window_height - 100))
        self.screen.blit(right_text, (self.window_width // 2 + 200, self.window_height - 100))
        
        # Update Display
        pygame.display.flip()
        self.clock.tick(30)
        
        return True
