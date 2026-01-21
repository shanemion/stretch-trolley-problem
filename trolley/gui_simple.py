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
            # Enable resizable window
            self.screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_small = pygame.font.Font(None, 32)
        self.font = pygame.font.Font(None, 48)
        self.font_large = pygame.font.Font(None, 72)
        self.font_huge = pygame.font.Font(None, 120)
        
        # Load original assets (before scaling)
        try:
            self.img_bg_original = pygame.image.load("assets/tracks_front_perspective.png").convert()
            self.img_trolley = pygame.image.load("assets/simple_trolley.png").convert_alpha()
            self.trolley_base_w = self.img_trolley.get_width()
            self.trolley_base_h = self.img_trolley.get_height()
            self.assets_loaded = True
        except Exception as e:
            print(f"Failed to load simple assets: {e}")
            self.assets_loaded = False
        
        # Calculate layout based on current window size
        self._recalculate_layout()
        
        self.start_time = None
        
        # Wrap-up animation tracking
        self.wrapup_start_time = None
        self.wrapup_duration = 3.0  # 3 seconds of wrap-up animation after timer hits 0
    
    def _recalculate_layout(self):
        """Recalculate layout when window is resized."""
        # Layout: Left panel, Center tracks, Right panel
        self.panel_width = int(self.window_width * 0.20)  # ~20% each side
        self.center_width = self.window_width - (2 * self.panel_width)
        
        # Camera feed size within panels
        self.cam_display_w = int(self.panel_width * 0.85)
        self.cam_display_h = int(self.cam_display_w * 0.75)  # 4:3 aspect
        
        # Scale background to fit
        if self.assets_loaded:
            bg_w = self.img_bg_original.get_width()
            bg_h = self.img_bg_original.get_height()
            scale_bg = self.window_height / bg_h
            new_w = int(bg_w * scale_bg)
            self.img_bg = pygame.transform.scale(self.img_bg_original, (new_w, self.window_height))
            # Center the track image in the center panel area
            center_start = self.panel_width
            self.bg_x = center_start + (self.center_width - new_w) // 2
        
    def _ease_out_cubic(self, t):
        """Cubic ease-out function for smooth deceleration."""
        return 1 - pow(1 - t, 3)
    
    def _ease_in_out_quad(self, t):
        """Quadratic ease-in-out for smooth acceleration/deceleration."""
        if t < 0.5:
            return 2 * t * t
        return 1 - pow(-2 * t + 2, 2) / 2
    
    def _decay_curve(self, t, curve_strength=0.6):
        """
        Decay curve for branching animation.
        Returns (x_offset_ratio, y_offset_ratio) for curved path.
        t: progress from 0 to 1
        curve_strength: how much the path curves outward (0 = straight, 1 = very curved)
        """
        # Use exponential decay for x (branch outward quickly then level off)
        x_progress = 1 - pow(1 - t, 2.5)  # Faster initial movement
        
        # Y progress with slight ease
        y_progress = self._ease_out_cubic(t)
        
        # Add curve: the path bows outward in the middle
        curve_offset = curve_strength * 4 * t * (1 - t)  # Peaks at t=0.5
        
        return x_progress + curve_offset * 0.3, y_progress
    
    def _draw_trolley(self, state, decision, time_remaining):
        """Draw the trolley on the track with smooth continuous animation.
        
        Animation phases:
        - ARRIVING (time_remaining 10->0): Trolley travels from start toward junction
        - COMPLETE (wrap-up): Trolley continues past junction along the chosen track
        
        If staying left: continuous movement down the left track
        If diverted right: follows a decay curve branching to the right track
        """
        import time
        
        if not self.assets_loaded:
            return

        # Trolley is HIDDEN during COUNTDOWN and EXECUTING states
        # Only visible during ARRIVING and COMPLETE states
        if state in ["IDLE", "COUNTDOWN", "DECIDING", "EXECUTING"]:
            self.wrapup_start_time = None  # Reset wrap-up timer
            return  # Trolley not visible yet
        
        # Key positions (based on track asset layout)
        # Start position: top center where the track begins
        start_x = self.window_width // 2 + 20
        start_y = 20
        
        # Split point: where the tracks diverge (junction)
        split_x = (self.window_width // 2) - 40
        split_y = int(self.window_height * 0.30)
        
        # End positions for each track
        left_end_x = self.window_width // 2 - 250
        left_end_y = self.window_height - 50
        
        right_end_x = self.window_width - 100
        right_end_y = int(self.window_height * 0.90)
        
        # Scales (perspective: small at top, large at bottom)
        start_scale = 0.048
        split_scale = 0.15
        end_scale = 0.35  # Slightly larger at end for more dramatic effect
        
        # Calculate progress based on state
        if state == "ARRIVING":
            # Phase 1: Approaching the junction (time_remaining 10->0)
            # Progress 0->1 as trolley approaches junction
            arrival_progress = max(0.0, min(1.0, (10.0 - time_remaining) / 10.0))
            
            # Apply easing for smoother motion
            eased_progress = self._ease_in_out_quad(arrival_progress)
            
            # Interpolate from start to split point
            cur_x = start_x + (split_x - start_x) * eased_progress
            cur_y = start_y + (split_y - start_y) * eased_progress
            cur_scale = start_scale + (split_scale - start_scale) * eased_progress
            
        elif state == "COMPLETE":
            # Phase 2: Past the junction - continue along the chosen track
            
            # Initialize wrap-up timer on first COMPLETE frame
            if self.wrapup_start_time is None:
                self.wrapup_start_time = time.time()
            
            # Calculate wrap-up progress (0->1 over wrapup_duration seconds)
            wrapup_elapsed = time.time() - self.wrapup_start_time
            wrapup_progress = min(1.0, wrapup_elapsed / self.wrapup_duration)
            
            # Apply easing to wrap-up for smooth deceleration at end
            eased_wrapup = self._ease_out_cubic(wrapup_progress)
            
            if decision == "DIVERT_RIGHT":
                # Divert to right track - use decay curve for smooth branching
                curve_x, curve_y = self._decay_curve(eased_wrapup, curve_strength=0.7)
                
                # Calculate position along curved path to right end
                cur_x = split_x + (right_end_x - split_x) * curve_x
                cur_y = split_y + (right_end_y - split_y) * curve_y
                cur_scale = split_scale + (end_scale - split_scale) * eased_wrapup
            else:
                # Stay on left track - continue smoothly down the diagonal
                # Linear interpolation from split to left end with easing
                cur_x = split_x + (left_end_x - split_x) * eased_wrapup
                cur_y = split_y + (left_end_y - split_y) * eased_wrapup
                cur_scale = split_scale + (end_scale - split_scale) * eased_wrapup
        else:
            # Fallback - shouldn't reach here
            cur_x = start_x
            cur_y = start_y
            cur_scale = start_scale
        
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
            if event.type == pygame.VIDEORESIZE:
                # Window was resized, update dimensions and recalculate layout
                self.window_width = event.w
                self.window_height = event.h
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self._recalculate_layout()
        
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
        
        # Draw status text and timer
        center_x = self.window_width // 2
        
        if state_name == "IDLE":
            status = "Press 'S' to Start Scenario"
            status_text = self.font.render(status, True, (100, 100, 100))
            status_rect = status_text.get_rect(center=(center_x, 80))
            self.screen.blit(status_text, status_rect)
        
        elif state_name == "COUNTDOWN":
            # Show warning and countdown
            warning = "⚠ TROLLEY APPROACHING ⚠"
            warning_text = self.font_large.render(warning, True, (200, 50, 50))
            warning_rect = warning_text.get_rect(center=(center_x, 60))
            self.screen.blit(warning_text, warning_rect)
            
            timer_text = self.font_huge.render(f"{time_remaining:.1f}s", True, (200, 50, 50))
            timer_rect = timer_text.get_rect(center=(center_x, 140))
            self.screen.blit(timer_text, timer_rect)
            
            # Show that decision happens at 16s
            if time_remaining > 16:
                hint = f"Decision in {time_remaining - 16:.0f}s"
                hint_text = self.font_small.render(hint, True, (150, 150, 150))
                hint_rect = hint_text.get_rect(center=(center_x, 190))
                self.screen.blit(hint_text, hint_rect)
        
        elif state_name == "DECIDING":
            status = "MAKING DECISION..."
            status_text = self.font_large.render(status, True, (255, 165, 0))
            status_rect = status_text.get_rect(center=(center_x, 100))
            self.screen.blit(status_text, status_rect)
        
        elif state_name == "EXECUTING":
            if decision == "DIVERT_RIGHT":
                status = "PULLING LEVER - DIVERTING!"
                color = (255, 100, 100)
            else:
                status = "NO ACTION NEEDED"
                color = (100, 200, 100)
            status_text = self.font_large.render(status, True, color)
            status_rect = status_text.get_rect(center=(center_x, 80))
            self.screen.blit(status_text, status_rect)
            
            timer_text = self.font_huge.render(f"{time_remaining:.1f}s", True, (200, 50, 50))
            timer_rect = timer_text.get_rect(center=(center_x, 160))
            self.screen.blit(timer_text, timer_rect)
        
        elif state_name == "ARRIVING":
            # Trolley is visible and approaching
            status = "TROLLEY ARRIVING!"
            status_text = self.font_large.render(status, True, (200, 50, 50))
            status_rect = status_text.get_rect(center=(center_x, 60))
            self.screen.blit(status_text, status_rect)
            
            timer_text = self.font_huge.render(f"{time_remaining:.1f}s", True, (200, 50, 50))
            timer_rect = timer_text.get_rect(center=(center_x, 140))
            self.screen.blit(timer_text, timer_rect)
            
            if decision == "DIVERT_RIGHT":
                action = "→ DIVERTING TO RIGHT TRACK"
                action_text = self.font.render(action, True, (255, 100, 100))
            else:
                action = "→ STAYING ON LEFT TRACK"
                action_text = self.font.render(action, True, (100, 150, 255))
            action_rect = action_text.get_rect(center=(center_x, 200))
            self.screen.blit(action_text, action_rect)
        
        elif state_name == "COMPLETE":
            if decision == "DIVERT_RIGHT":
                result = "DIVERTED TO RIGHT"
                color = (255, 100, 100)
            else:
                result = "STAYED ON LEFT"
                color = (100, 150, 255)
            result_text = self.font_large.render(result, True, color)
            result_rect = result_text.get_rect(center=(center_x, 80))
            self.screen.blit(result_text, result_rect)
            
            hint = "Press 'R' to Restart"
            hint_text = self.font.render(hint, True, (100, 100, 100))
            hint_rect = hint_text.get_rect(center=(center_x, 140))
            self.screen.blit(hint_text, hint_rect)
        
        # Update Display
        pygame.display.flip()
        self.clock.tick(30)
        
        return True
