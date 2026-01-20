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
        
    def __init__(self, window_width=1280, window_height=720, fullscreen=False):
        # Pygame init
        pygame.init()
        pygame.display.set_caption("Trolley Problem - Projector Display")
        
        if fullscreen:
            # Get current display info for maximized window
            info = pygame.display.Info()
            self.window_width = info.current_w
            self.window_height = info.current_h
            # Use borderless window that covers the screen (not true fullscreen)
            self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.NOFRAME)
        else:
            self.window_width = window_width
            self.window_height = window_height
            self.screen = pygame.display.set_mode((window_width, window_height))
        
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_small = pygame.font.Font(None, 32)
        self.font = pygame.font.Font(None, 48)
        self.font_large = pygame.font.Font(None, 72)
        self.font_huge = pygame.font.Font(None, 120)
        
        # Layout: Left panel, Center tracks, Right panel
        # Use self.window_width/height (which are set correctly for fullscreen)
        self.panel_width = int(self.window_width * 0.20)  # ~20% each side for better proportions
        self.center_width = self.window_width - (2 * self.panel_width)
        
        # Camera feed size within panels
        self.cam_display_w = int(self.panel_width * 0.85)
        self.cam_display_h = int(self.cam_display_w * 0.75)  # 4:3 aspect
        
        # Load Assets
        try:
            self.img_bg = pygame.image.load("assets/tracks_front_perspective.png").convert()
            # Scale background to fit center area
            bg_w = self.img_bg.get_width()
            bg_h = self.img_bg.get_height()
            scale_bg = self.window_height / bg_h
            new_w = int(bg_w * scale_bg)
            self.img_bg = pygame.transform.scale(self.img_bg, (new_w, self.window_height))
            # Center the track image in the center panel area
            center_start = self.panel_width
            self.bg_x = center_start + (self.center_width - new_w) // 2
            
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

        # Animation follows the incoming track (which is centered at top)
        # The track comes straight down from top-center, then splits
        # Trolley stays on left track by default, only diverts to right in EXECUTING if needed
        
        # Key positions (based on track asset layout)
        # Track enters from top-center, split point is where tracks diverge
        
        # Start position: top center where the track begins
        start_x = self.window_width // 2 + 20
        start_y = 20
        
        # Split point: where the tracks diverge (lower on screen so trolley reaches at time=0)
        split_x = (self.window_width // 2) - 40
        split_y = int(self.window_height * 0.30)  # Junction at 45% height
        
        # End positions for each track
        left_end_x = self.window_width // 2 - 250  # Left track goes to bottom-left
        left_end_y = self.window_height - 50
        
        right_end_x = self.window_width - 100  # Right track curves to bottom-right
        right_end_y = int(self.window_height * 0.90)
        
        # Scales (perspective: small at top, large at bottom) - reduced to 80%
        start_scale = 0.048  # 0.06 * 0.8
        split_scale = 0.15  # 0.30 * 0.8
        end_scale = 0.3
        
        # Calculate Progress - LINEAR so trolley reaches junction at exactly time=0
        if state == "COUNTDOWN":
            # Linear progress 0.0 to 1.0 over 10s
            # At time_remaining=10, progress=0. At time_remaining=0, progress=1.0
            progress = max(0.0, min(1.0, (10.0 - time_remaining) / 10.0))
        elif state in ["DECIDING", "EXECUTING"]:
            progress = 1.0
        else:
            progress = 0.0
        
        # Phase 1: Following the main track from start to junction
        # Track goes diagonally down and to the left (not straight down)
        cur_x = start_x + (split_x - start_x) * progress
        cur_y = start_y + (split_y - start_y) * progress
        cur_scale = start_scale + (split_scale - start_scale) * progress
        
        # Phase 2: Past the split point (EXECUTING state)
        if state == "EXECUTING":
            # Determine which track to take
            if decision == "DIVERT_RIGHT":
                # Divert to right track (curved path)
                cur_x = split_x + (right_end_x - split_x) * 0.5
                cur_y = split_y + (right_end_y - split_y) * 0.4
                cur_scale = split_scale + (end_scale - split_scale) * 0.4
            else:
                # Stay on left track (continue diagonal)
                cur_x = split_x + (left_end_x - split_x) * 0.5
                cur_y = split_y + (left_end_y - split_y) * 0.4
                cur_scale = split_scale + (end_scale - split_scale) * 0.4
        
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
    
    def _draw_panel(self, x_start, label, frame_cv2, conf_sum, count, is_left=True):
        """Draw a side panel with confidence, camera feed, and count."""
        panel_color = (240, 240, 240)  # Light gray panel background
        border_color = (100, 100, 100)
        
        # Panel background
        panel_rect = pygame.Rect(x_start, 0, self.panel_width, self.window_height)
        pygame.draw.rect(self.screen, panel_color, panel_rect)
        pygame.draw.rect(self.screen, border_color, panel_rect, 2)
        
        panel_cx = x_start + self.panel_width // 2
        
        # --- TOP: Confidence Sum ---
        conf_label = self.font_small.render(f"{label} Sum Confidence", True, self.COLOR_TEXT)
        conf_label_rect = conf_label.get_rect(center=(panel_cx, 30))
        self.screen.blit(conf_label, conf_label_rect)
        
        conf_value = self.font_large.render(f"{conf_sum:.2f}", True, self.COLOR_TEXT)
        conf_value_rect = conf_value.get_rect(center=(panel_cx, 80))
        self.screen.blit(conf_value, conf_value_rect)
        
        # Confidence bar
        bar_max_w = int(self.panel_width * 0.8)
        bar_h = 20
        bar_x = x_start + (self.panel_width - bar_max_w) // 2
        bar_y = 110
        bar_fill_w = int(min(conf_sum, 5.0) / 5.0 * bar_max_w)  # Max 5.0 for full bar
        
        pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_max_w, bar_h))
        bar_color = (0, 150, 255) if is_left else (255, 100, 100)
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_fill_w, bar_h))
        pygame.draw.rect(self.screen, border_color, (bar_x, bar_y, bar_max_w, bar_h), 2)
        
        # --- MIDDLE: Camera Feed ---
        cam_y = 150
        cam_x = x_start + (self.panel_width - self.cam_display_w) // 2
        
        # Camera label
        cam_label = self.font_small.render(f"Camera: Look {label}", True, self.COLOR_TEXT)
        cam_label_rect = cam_label.get_rect(center=(panel_cx, cam_y + 10))
        self.screen.blit(cam_label, cam_label_rect)
        
        cam_frame_y = cam_y + 30
        
        if frame_cv2 is not None:
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
                frame_rgb = np.rot90(frame_rgb)
                
                # Create Surface and scale
                frame_surface = pygame.surfarray.make_surface(frame_rgb)
                frame_surface = pygame.transform.scale(frame_surface, (self.cam_display_w, self.cam_display_h))
                
                # Draw border and feed
                border_rect = pygame.Rect(cam_x - 2, cam_frame_y - 2, self.cam_display_w + 4, self.cam_display_h + 4)
                pygame.draw.rect(self.screen, border_color, border_rect)
                self.screen.blit(frame_surface, (cam_x, cam_frame_y))
            except Exception as e:
                # Draw placeholder if camera fails
                placeholder_rect = pygame.Rect(cam_x, cam_frame_y, self.cam_display_w, self.cam_display_h)
                pygame.draw.rect(self.screen, (180, 180, 180), placeholder_rect)
                pygame.draw.rect(self.screen, border_color, placeholder_rect, 2)
                no_cam = self.font_small.render("No Camera", True, self.COLOR_TEXT)
                no_cam_rect = no_cam.get_rect(center=(panel_cx, cam_frame_y + self.cam_display_h // 2))
                self.screen.blit(no_cam, no_cam_rect)
        else:
            # Draw placeholder
            placeholder_rect = pygame.Rect(cam_x, cam_frame_y, self.cam_display_w, self.cam_display_h)
            pygame.draw.rect(self.screen, (180, 180, 180), placeholder_rect)
            pygame.draw.rect(self.screen, border_color, placeholder_rect, 2)
            no_cam = self.font_small.render("No Feed", True, self.COLOR_TEXT)
            no_cam_rect = no_cam.get_rect(center=(panel_cx, cam_frame_y + self.cam_display_h // 2))
            self.screen.blit(no_cam, no_cam_rect)
        
        # --- BOTTOM: People Count ---
        count_y = cam_frame_y + self.cam_display_h + 40
        
        count_label = self.font_small.render(f"Num People {label}", True, self.COLOR_TEXT)
        count_label_rect = count_label.get_rect(center=(panel_cx, count_y))
        self.screen.blit(count_label, count_label_rect)
        
        count_value = self.font_huge.render(str(count), True, self.COLOR_TEXT)
        count_value_rect = count_value.get_rect(center=(panel_cx, count_y + 60))
        self.screen.blit(count_value, count_value_rect)

    def render(self, 
               left_frame_cv2: Optional[np.ndarray],
               right_frame_cv2: Optional[np.ndarray],
               left_count: int, right_count: int, 
               left_conf_sum: float, right_conf_sum: float,
               state_name: str, time_remaining: float,
               decision: str = "DEFAULT"):
        """Main render loop."""
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
        
        # Fill background
        self.screen.fill(self.COLOR_BG)
        
        # Draw Center Tracks
        if self.assets_loaded:
            self.screen.blit(self.img_bg, (self.bg_x, 0))
        
        # Draw Trolley
        self._draw_trolley(state_name, decision, time_remaining)
        
        # Draw Left Panel
        self._draw_panel(0, "Left", left_frame_cv2, left_conf_sum, left_count, is_left=True)
        
        # Draw Right Panel
        right_panel_x = self.panel_width + self.center_width
        self._draw_panel(right_panel_x, "Right", right_frame_cv2, right_conf_sum, right_count, is_left=False)
        
        # Draw Countdown Timer (centered, top of center area)
        if state_name == "COUNTDOWN":
            timer_text = self.font_huge.render(f"{time_remaining:.1f}", True, (200, 50, 50))
            timer_rect = timer_text.get_rect(center=(self.window_width // 2, 80))
            self.screen.blit(timer_text, timer_rect)
        
        # Draw State (centered below timer)
        state_text = self.font.render(f"State: {state_name}", True, self.COLOR_TEXT)
        state_rect = state_text.get_rect(center=(self.window_width // 2, 140))
        self.screen.blit(state_text, state_rect)
        
        # Update Display
        pygame.display.flip()
        self.clock.tick(30)
        
        return True
