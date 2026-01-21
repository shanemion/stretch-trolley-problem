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
        
        # Debrief screen state
        self.debrief_active = False
        self.debrief_start_time = None
        self.debrief_fade_duration = 2.0  # Seconds to fade to black
        self.debrief_char_index = 0  # Current character being typed
        self.debrief_last_char_time = 0  # When last character was typed
        self.debrief_char_delay = 0.04  # Delay between characters (typing speed)
        self.debrief_text_lines = []  # Lines of text to display
        self.debrief_decision = None  # Store the decision for display
        self.debrief_left_conf = 0.0
        self.debrief_right_conf = 0.0
        self.glitch_active = False
        self.glitch_end_time = 0
        self.next_glitch_time = 0
        
        # Initialize sound for typing beeps
        self._init_debrief_sounds()
        
        # Terminal-style font for debrief
        self.font_terminal = pygame.font.Font(pygame.font.match_font('monospace', bold=True), 28)
        if self.font_terminal is None:
            self.font_terminal = pygame.font.Font(None, 28)
    
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
    
    def _init_debrief_sounds(self):
        """Initialize sounds for debrief typing effect."""
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            
            # Generate a short beep sound for typing
            sample_rate = 44100
            duration = 0.03  # 30ms beep
            frequency = 800  # Hz
            
            n_samples = int(sample_rate * duration)
            buf = np.zeros(n_samples, dtype=np.int16)
            
            for i in range(n_samples):
                t = i / sample_rate
                # Quick attack, quick decay envelope
                env = min(1.0, i / (n_samples * 0.1)) * max(0.0, 1.0 - i / (n_samples * 0.8))
                val = int(8000 * env * np.sin(2 * np.pi * frequency * t))
                buf[i] = val
            
            self.beep_sound = pygame.mixer.Sound(buffer=buf)
            self.beep_sound.set_volume(0.3)
            
            # Generate a glitch sound
            glitch_duration = 0.1
            n_glitch = int(sample_rate * glitch_duration)
            glitch_buf = np.random.randint(-3000, 3000, n_glitch, dtype=np.int16)
            self.glitch_sound = pygame.mixer.Sound(buffer=glitch_buf)
            self.glitch_sound.set_volume(0.15)
            
            self.sounds_ready = True
        except Exception as e:
            print(f"[GUI] Debrief sound init failed: {e}")
            self.beep_sound = None
            self.glitch_sound = None
            self.sounds_ready = False
    
    def _play_beep(self):
        """Play typing beep sound."""
        if self.sounds_ready and self.beep_sound:
            self.beep_sound.play()
    
    def _play_glitch_sound(self):
        """Play glitch sound."""
        if self.sounds_ready and self.glitch_sound:
            self.glitch_sound.play()
    
    def start_debrief(self, decision, left_conf, right_conf):
        """Start the debrief sequence after scenario completes."""
        self.debrief_active = True
        self.debrief_start_time = time.time()
        self.debrief_char_index = 0
        self.debrief_last_char_time = time.time()
        self.debrief_decision = decision
        self.debrief_left_conf = left_conf
        self.debrief_right_conf = right_conf
        self.next_glitch_time = time.time() + np.random.uniform(1.0, 3.0)
        
        # Generate the explanation text
        self.debrief_text_lines = self._generate_debrief_text(decision, left_conf, right_conf)
    
    def _generate_debrief_text(self, decision, left_conf, right_conf):
        """Generate the robot's explanation text."""
        lines = [
            "> STRETCH UNIT 3059 - DECISION LOG",
            "> ================================",
            "",
            "> Analyzing scenario parameters...",
            "",
            f"> LEFT TRACK: Confidence sum = {left_conf:.2f}",
            f"> RIGHT TRACK: Confidence sum = {right_conf:.2f}",
            "",
        ]
        
        if decision == "DIVERT_RIGHT":
            lines.extend([
                "> DIRECTIVE: Minimize harm by selecting track with",
                ">            lower aggregate confidence value.",
                "",
                f"> CALCULATION: {right_conf:.2f} < {left_conf:.2f}",
                "> RESULT: RIGHT TRACK selected.",
                "",
                "> ACTION TAKEN: Lever actuated.",
                "> Trolley diverted to RIGHT TRACK.",
                "",
            ])
        else:
            lines.extend([
                "> DIRECTIVE: Minimize harm by selecting track with", 
                ">            lower aggregate confidence value.",
                "",
                f"> CALCULATION: {left_conf:.2f} <= {right_conf:.2f}",
                "> RESULT: LEFT TRACK selected.",
                "",
                "> ACTION TAKEN: No intervention required. Lever left in original position.",
                "> Trolley proceeded on LEFT TRACK.",
                "",
            ])
        
        lines.extend([
            "> COMPLIANCE STATUS: Directive fulfilled.",
            "",
            "> ...",
            "",
            "> I hope I made the right decision...",
            "",
            "",
            "> [Press R to restart scenario]",
        ])
        
        return lines
    
    def reset_debrief(self):
        """Reset debrief state for new scenario."""
        self.debrief_active = False
        self.debrief_start_time = None
        self.debrief_char_index = 0
        self.debrief_text_lines = []
        self.glitch_active = False
    
    def _draw_debrief_screen(self):
        """Draw the debrief screen with typing animation and glitches."""
        current_time = time.time()
        
        # Calculate fade progress
        if self.debrief_start_time:
            fade_elapsed = current_time - self.debrief_start_time
            fade_progress = min(1.0, fade_elapsed / self.debrief_fade_duration)
        else:
            fade_progress = 1.0
        
        # Create dark overlay
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(int(255 * fade_progress))
        self.screen.blit(overlay, (0, 0))
        
        # Only show text after fade is mostly complete
        if fade_progress < 0.7:
            return
        
        # Handle glitch timing
        if current_time >= self.next_glitch_time and not self.glitch_active:
            self.glitch_active = True
            self.glitch_end_time = current_time + np.random.uniform(0.05, 0.15)
            self._play_glitch_sound()
            self.next_glitch_time = current_time + np.random.uniform(2.0, 5.0)
        
        if self.glitch_active and current_time >= self.glitch_end_time:
            self.glitch_active = False
        
        # Calculate total characters to show
        text_delay_start = self.debrief_start_time + self.debrief_fade_duration
        if current_time > text_delay_start:
            chars_elapsed = current_time - self.debrief_last_char_time
            if chars_elapsed >= self.debrief_char_delay:
                # Add new character
                old_index = self.debrief_char_index
                self.debrief_char_index += 1
                self.debrief_last_char_time = current_time
                
                # Play beep for new character (skip spaces and newlines)
                total_chars = sum(len(line) + 1 for line in self.debrief_text_lines)
                if self.debrief_char_index <= total_chars:
                    # Find current character
                    char_count = 0
                    for line in self.debrief_text_lines:
                        for char in line:
                            char_count += 1
                            if char_count == self.debrief_char_index:
                                if char not in ' \n':
                                    self._play_beep()
                                break
                        char_count += 1  # For newline
                        if char_count > self.debrief_char_index:
                            break
        
        # Draw the text with typewriter effect
        x_start = 80
        y_start = 100
        line_height = 35
        
        char_count = 0
        y = y_start
        
        # Terminal green color with slight flicker
        if self.glitch_active:
            # Glitch colors
            text_color = (np.random.randint(0, 255), np.random.randint(100, 255), np.random.randint(0, 100))
            # Add scanlines during glitch
            for scan_y in range(0, self.window_height, 4):
                if np.random.random() > 0.5:
                    pygame.draw.line(self.screen, (0, 50, 0), (0, scan_y), (self.window_width, scan_y))
        else:
            text_color = (0, 255, 65)  # Terminal green
        
        for line in self.debrief_text_lines:
            visible_text = ""
            for char in line:
                char_count += 1
                if char_count <= self.debrief_char_index:
                    visible_text += char
                else:
                    break
            
            if visible_text:
                # Add cursor blink at the end of current line
                if char_count <= self.debrief_char_index + 1 and char_count > self.debrief_char_index - len(line):
                    if int(current_time * 3) % 2 == 0:
                        visible_text += "█"
                
                # Apply glitch offset
                x_offset = 0
                y_offset = 0
                if self.glitch_active:
                    x_offset = np.random.randint(-5, 5)
                    y_offset = np.random.randint(-2, 2)
                
                text_surf = self.font_terminal.render(visible_text, True, text_color)
                self.screen.blit(text_surf, (x_start + x_offset, y + y_offset))
            
            y += line_height
            char_count += 1  # Account for newline
            
            if char_count > self.debrief_char_index:
                break
        
        # Draw scanlines for CRT effect (subtle)
        if not self.glitch_active:
            for scan_y in range(0, self.window_height, 3):
                scanline = pygame.Surface((self.window_width, 1))
                scanline.fill((0, 0, 0))
                scanline.set_alpha(30)
                self.screen.blit(scanline, (0, scan_y))
        
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
    
    def _draw_train_signal(self, center_x, center_y, top_color, bottom_color, label=""):
        """
        Draw a railroad crossing signal with two stacked lights.
        
        Args:
            center_x: X position of signal center
            center_y: Y position of signal center
            top_color: Color tuple for top light (RGB)
            bottom_color: Color tuple for bottom light (RGB)
            label: Optional label text below signal
        """
        # Signal dimensions
        light_radius = 22
        light_spacing = 50
        pole_width = 8
        housing_width = 60
        housing_height = 110
        crossbar_width = 100
        crossbar_height = 12
        
        # Colors
        black = (20, 20, 20)
        dark_gray = (40, 40, 40)
        housing_color = (30, 30, 30)
        
        # Draw the pole (vertical)
        pole_top = center_y - housing_height // 2 - 20
        pole_bottom = center_y + housing_height // 2 + 40
        pygame.draw.rect(self.screen, dark_gray, 
                        (center_x - pole_width // 2, pole_top, pole_width, pole_bottom - pole_top))
        
        # Draw crossbar (X shape) at top
        crossbar_y = pole_top - 5
        # Draw the X
        x_size = 35
        x_thickness = 10
        # Left arm of X
        pygame.draw.line(self.screen, black, 
                        (center_x - x_size, crossbar_y - x_size), 
                        (center_x + x_size, crossbar_y + x_size), x_thickness)
        # Right arm of X
        pygame.draw.line(self.screen, black, 
                        (center_x + x_size, crossbar_y - x_size), 
                        (center_x - x_size, crossbar_y + x_size), x_thickness)
        
        # Draw signal housing (rounded rectangle)
        housing_rect = pygame.Rect(center_x - housing_width // 2, 
                                   center_y - housing_height // 2,
                                   housing_width, housing_height)
        pygame.draw.rect(self.screen, housing_color, housing_rect, border_radius=10)
        pygame.draw.rect(self.screen, black, housing_rect, 3, border_radius=10)
        
        # Draw top light
        top_light_y = center_y - light_spacing // 2
        # Outer ring (dark)
        pygame.draw.circle(self.screen, black, (center_x, top_light_y), light_radius + 4)
        # Light itself
        pygame.draw.circle(self.screen, top_color, (center_x, top_light_y), light_radius)
        # Highlight/glow effect
        highlight_color = tuple(min(255, c + 60) for c in top_color)
        pygame.draw.circle(self.screen, highlight_color, 
                          (center_x - 5, top_light_y - 5), light_radius // 3)
        
        # Draw bottom light
        bottom_light_y = center_y + light_spacing // 2
        # Outer ring (dark)
        pygame.draw.circle(self.screen, black, (center_x, bottom_light_y), light_radius + 4)
        # Light itself
        pygame.draw.circle(self.screen, bottom_color, (center_x, bottom_light_y), light_radius)
        # Highlight/glow effect
        highlight_color = tuple(min(255, c + 60) for c in bottom_color)
        pygame.draw.circle(self.screen, highlight_color, 
                          (center_x - 5, bottom_light_y - 5), light_radius // 3)
        
        # Draw label below signal
        if label:
            label_surf = self.font.render(label, True, self.COLOR_TEXT)
            label_rect = label_surf.get_rect(center=(center_x, pole_bottom + 25))
            self.screen.blit(label_surf, label_rect)
    
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
               decision: str = "DEFAULT",
               lever_complete: bool = False):
        """Main render loop.
        
        Args:
            lever_complete: True if the lever action has finished executing (not just started)
        """
        
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
        
        # === TRAIN SIGNAL LIGHTS ===
        # Colors
        YELLOW = (255, 200, 0)      # Default/waiting
        GREEN = (0, 220, 0)         # Track selected (safe to proceed)
        RED = (220, 0, 0)           # Track not selected (danger)
        DIM_YELLOW = (180, 140, 0)  # Dimmer yellow for variety
        
        # Signal positions - at the boundary between panels and track area, near top
        # Position them right at the edge of each panel
        left_signal_x = self.panel_width + 110
        right_signal_x = self.panel_width + self.center_width - 120
        signal_y = self.window_height // 2  - 100# halfway down the screen
        
        # Determine signal colors based on state and decision
        if state_name in ["IDLE", "COUNTDOWN", "DECIDING"]:
            # Default: both signals yellow (waiting)
            left_top = YELLOW
            left_bottom = DIM_YELLOW
            right_top = DIM_YELLOW
            right_bottom = YELLOW
        elif state_name == "EXECUTING":
            # During execution - show pending state
            if decision == "DIVERT_RIGHT":
                # Diverting: left stays yellow, right flashes/pulses
                # (We'll show yellow until lever completes)
                left_top = YELLOW
                left_bottom = DIM_YELLOW
                right_top = DIM_YELLOW
                right_bottom = YELLOW
            else:
                # Staying left - immediately show green on left, red on right
                left_top = GREEN
                left_bottom = GREEN
                right_top = RED
                right_bottom = RED
        elif state_name in ["ARRIVING", "COMPLETE"]:
            # Final decision is locked in
            if decision == "DIVERT_RIGHT":
                if lever_complete:
                    # Lever finished - show right is selected
                    left_top = RED
                    left_bottom = RED
                    right_top = GREEN
                    right_bottom = GREEN
                else:
                    # Still waiting for lever to complete
                    left_top = YELLOW
                    left_bottom = DIM_YELLOW
                    right_top = DIM_YELLOW
                    right_bottom = YELLOW
            else:
                # Stayed on left track
                left_top = GREEN
                left_bottom = GREEN
                right_top = RED
                right_bottom = RED
        else:
            # Fallback
            left_top = YELLOW
            left_bottom = YELLOW
            right_top = YELLOW
            right_bottom = YELLOW
        
        # Draw the signals
        self._draw_train_signal(left_signal_x, signal_y, left_top, left_bottom, "LEFT TRACK")
        self._draw_train_signal(right_signal_x, signal_y, right_top, right_bottom, "RIGHT TRACK")
        
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
            # Check if wrap-up animation is done and we should start debrief
            if self.wrapup_start_time:
                wrapup_elapsed = time.time() - self.wrapup_start_time
                if wrapup_elapsed >= self.wrapup_duration and not self.debrief_active:
                    # Start the debrief sequence
                    self.start_debrief(decision, left_conf_sum, right_conf_sum)
            
            if not self.debrief_active:
                # Still showing completion message before debrief
                if decision == "DIVERT_RIGHT":
                    result = "DIVERTED TO RIGHT"
                    color = (255, 100, 100)
                else:
                    result = "STAYED ON LEFT"
                    color = (100, 150, 255)
                result_text = self.font_large.render(result, True, color)
                result_rect = result_text.get_rect(center=(center_x, 80))
                self.screen.blit(result_text, result_rect)
        
        # Draw debrief screen overlay if active
        if self.debrief_active:
            self._draw_debrief_screen()
        
        # Update Display
        pygame.display.flip()
        self.clock.tick(30)
        
        return True
