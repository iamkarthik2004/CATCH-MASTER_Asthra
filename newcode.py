import pygame
import random
import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("game_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

class GameConfig:
    """Configuration for game settings"""
    
    def __init__(self):
        # Display settings
        self.width = 800
        self.height = 600
        self.fps = 60
        self.fullscreen = False
        
        # Game mechanics
        self.difficulty = "normal"  # easy, normal, hard
        self.ball_spawn_intervals = {"easy": 2000, "normal": 1500, "hard": 1000}
        self.ball_speeds = {"easy": 4, "normal": 6, "hard": 8}
        self.ball_acceleration = 0.1  # Balls speed up over time
        
        # Controls
        self.control_mode = "hand"  # hand, keyboard, or auto (for testing)
        self.hand_sensitivity = 5
        self.keyboard_speed = 10
        
        # Colors
        self.colors = {
            "BLACK": (0, 0, 0),
            "WHITE": (255, 255, 255),
            "RED": (255, 0, 0),
            "GREEN": (0, 255, 0),
            "BLUE": (0, 0, 255),
            "YELLOW": (255, 255, 0),
            "PURPLE": (128, 0, 128),
            "CYAN": (0, 255, 255),
            "ORANGE": (255, 165, 0),
        }
        
        # Paths
        self.asset_dir = Path("assets")
        self.sound_dir = self.asset_dir / "sounds"
        self.image_dir = self.asset_dir / "images"
        
        # GPU settings
        self.use_gpu = True  # Will be automatically set based on availability
        
    def update_difficulty(self, difficulty):
        if difficulty in ["easy", "normal", "hard"]:
            self.difficulty = difficulty
            logging.info(f"Difficulty set to {difficulty}")
            return True
        return False
        
    def toggle_control_mode(self):
        modes = ["hand", "keyboard", "auto"]
        current_index = modes.index(self.control_mode)
        self.control_mode = modes[(current_index + 1) % len(modes)]
        logging.info(f"Control mode set to {self.control_mode}")
        return self.control_mode


class ResourceManager:
    """Handles loading and managing game resources"""
    
    def __init__(self, config):
        self.config = config
        self.fonts = {}
        self.sounds = {}
        self.images = {}
        
        # Make sure asset directories exist
        self.config.asset_dir.mkdir(exist_ok=True)
        self.config.sound_dir.mkdir(exist_ok=True)
        self.config.image_dir.mkdir(exist_ok=True)
        
        # Initialize default fonts
        self.fonts["small"] = pygame.font.SysFont(None, 24)
        self.fonts["medium"] = pygame.font.SysFont(None, 36)
        self.fonts["large"] = pygame.font.SysFont(None, 72)
        
    def get_font(self, size):
        return self.fonts.get(size, self.fonts["medium"])
        
    def render_text(self, text, size="medium", color=None):
        if color is None:
            color = self.config.colors["WHITE"]
        font = self.get_font(size)
        return font.render(text, True, color)


class HandDetector:
    """Handles hand detection using MediaPipe"""
    
    def __init__(self, config):
        self.config = config
        self.cap = None
        self.hands = None
        self.mp_hands = None
        self.mp_draw = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.detection_fps = 0
        self.last_positions = []  # Track last hand positions for smoothing
        
        # Initialize camera and MediaPipe
        self.initialize()
        
    def initialize(self):
        # Initialize OpenCV webcam
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logging.error("Failed to open webcam")
                return False
                
            # Try to set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logging.info("Webcam initialized")
        except Exception as e:
            logging.error(f"Webcam initialization error: {e}")
            return False
            
        # Initialize MediaPipe Hands
        try:
            self.mp_hands = mp.solutions.hands
            
            # Check if GPU acceleration is available for MediaPipe
            if self.config.use_gpu:
                try:
                    self.hands = self.mp_hands.Hands(
                        max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7,
                        model_complexity=1,
                        static_image_mode=False
                    )
                    logging.info("MediaPipe Hands initialized with GPU support")
                except Exception as gpu_error:
                    logging.warning(f"GPU acceleration failed: {gpu_error}. Falling back to CPU.")
                    self.config.use_gpu = False
                    self.hands = self.mp_hands.Hands(
                        max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5
                    )
            else:
                self.hands = self.mp_hands.Hands(
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                
            self.mp_draw = mp.solutions.drawing_utils
            logging.info("MediaPipe Hands initialized")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize MediaPipe Hands: {e}")
            if self.cap:
                self.cap.release()
            return False
            
    def get_smoothed_position(self, x, y):
        """Apply smoothing to hand position to prevent jitter"""
        self.last_positions.append((x, y))
        if len(self.last_positions) > 5:
            self.last_positions.pop(0)
            
        # Calculate average position
        avg_x = sum(pos[0] for pos in self.last_positions) / len(self.last_positions)
        avg_y = sum(pos[1] for pos in self.last_positions) / len(self.last_positions)
        
        return avg_x, avg_y
            
    def detect_gesture(self):
        """Detect hand gestures and return control information"""
        if not self.cap or not self.hands:
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Failed to read frame from webcam")
                return None
                
            # Calculate detection FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                self.detection_fps = self.frame_count / (current_time - self.fps_start_time)
                self.frame_count = 0
                self.fps_start_time = current_time
                
            # Process frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            rgb_frame.flags.writeable = False
            results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            # Default values
            direction = None
            hand_x = None
            hand_y = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness if results.multi_handedness else []
                ):
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    if handedness and handedness.classification:
                        label = handedness.classification[0].label
                    else:
                        label = "Unknown"
                    
                    wrist = hand_landmarks.landmark[0]
                    wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])
                    
                    index_tip = hand_landmarks.landmark[8]
                    finger_x, finger_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                    
                    smooth_x, smooth_y = self.get_smoothed_position(finger_x, finger_y)
                    finger_x, finger_y = int(smooth_x), int(smooth_y)
                    
                    cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 255), -1)
                    
                    hand_x = finger_x / frame.shape[1] * self.config.width
                    hand_y = finger_y / frame.shape[0] * self.config.height
                    
                    cv2.putText(
                        frame, f"{label} ({finger_x}, {finger_y})",
                        (wrist_x, wrist_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
            
            cv2.putText(
                frame, f"Detection FPS: {self.detection_fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            cv2.putText(
                frame, f"Control: {self.config.control_mode}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            cv2.imshow("Hand Detection Feed", frame)
            cv2.moveWindow("Hand Detection Feed", self.config.width + 10, 0)
            
            return {
                "hand_x": hand_x,
                "hand_y": hand_y,
            }
            
        except Exception as e:
            logging.error(f"Error in detect_hand_gesture: {e}")
            return None
            
    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


class Pot:
    """The player-controlled pot that catches balls"""
    
    def __init__(self, config):
        self.config = config
        self.width = 120
        self.height = 25
        self.x = config.width // 2 - self.width // 2
        self.y = config.height - 50
        self.speed = config.keyboard_speed
        self.target_x = self.x
        self.color = config.colors["GREEN"]
        self.trail = []
        
    def move_to_position(self, x):
        self.target_x = x - (self.width // 2)
        self.x += (self.target_x - self.x) * 0.3
        self.x = max(0, min(self.config.width - self.width, self.x))
        
    def move(self, direction):
        if direction == "left" and self.x > 0:
            self.x -= self.speed
        elif direction == "right" and self.x < self.config.width - self.width:
            self.x += self.speed
        self.x = max(0, min(self.config.width - self.width, self.x))
        
    def update(self):
        self.trail.append((self.x, self.y))
        if len(self.trail) > 10:
            self.trail.pop(0)
            
    def render(self, surface):
        for i, (trail_x, trail_y) in enumerate(self.trail):
            alpha = i / len(self.trail)
            trail_color = tuple(int(c * alpha) for c in self.color)
            trail_width = int(self.width * (0.7 + 0.3 * alpha))
            trail_height = int(self.height * (0.7 + 0.3 * alpha))
            x_offset = (self.width - trail_width) // 2
            pygame.draw.rect(
                surface,
                trail_color,
                [trail_x + x_offset, trail_y, trail_width, trail_height],
                border_radius=8
            )
        
        pygame.draw.rect(
            surface,
            self.color,
            [self.x, self.y, self.width, self.height],
            border_radius=8
        )
        
        pygame.draw.rect(
            surface,
            self.config.colors["WHITE"],
            [self.x, self.y, self.width, self.height],
            width=2,
            border_radius=8
        )
        
        highlight_rect = pygame.Rect(self.x + 5, self.y + 3, self.width - 10, 5)
        pygame.draw.rect(surface, (255, 255, 255, 128), highlight_rect, border_radius=3)


class Ball:
    """Falling ball that player needs to catch"""
    
    def __init__(self, config):
        self.config = config
        self.radius = random.randint(10, 20)
        self.x = random.randint(self.radius, config.width - self.radius)
        self.y = -self.radius
        
        base_speed = config.ball_speeds[config.difficulty]
        self.speed = base_speed * (1 + random.uniform(-0.2, 0.2))
        
        ball_types = ["normal", "bonus", "speed"]
        weights = [0.7, 0.2, 0.1]
        self.ball_type = random.choices(ball_types, weights=weights)[0]
        
        if self.ball_type == "normal":
            self.color = config.colors["RED"]
            self.points = 1
        elif self.ball_type == "bonus":
            self.color = config.colors["YELLOW"]
            self.points = 3
            self.radius += 5
        elif self.ball_type == "speed":
            self.color = config.colors["CYAN"]
            self.points = 2
            self.speed *= 1.5
            self.radius -= 3
            
        self.trail = []
        self.particles = []
        
    def update(self):
        previous_y = self.y
        self.y += self.speed
        self.speed += self.config.ball_acceleration
        
        if abs(self.y - previous_y) > 2:
            self.trail.append((self.x, self.y))
            if len(self.trail) > 5:
                self.trail.pop(0)
                
        for particle in self.particles[:]:
            particle["life"] -= 1
            if particle["life"] <= 0:
                self.particles.remove(particle)
            else:
                particle["x"] += particle["dx"]
                particle["y"] += particle["dy"]
                particle["dy"] += 0.1
                
    def create_particles(self, num_particles=10):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(1, 3)
            self.particles.append({
                "x": self.x,
                "y": self.y,
                "dx": speed * np.cos(angle),
                "dy": speed * np.sin(angle),
                "radius": random.randint(2, 5),
                "color": self.color,
                "life": random.randint(10, 30)
            })
        
    def render(self, surface):
        for i, (trail_x, trail_y) in enumerate(self.trail):
            alpha = i / len(self.trail) if self.trail else 0
            trail_color = tuple(int(c * alpha) for c in self.color)
            trail_radius = int(self.radius * (0.5 + 0.5 * alpha))
            pygame.draw.circle(surface, trail_color, (int(trail_x), int(trail_y)), trail_radius)
        
        for particle in self.particles:
            pygame.draw.circle(
                surface,
                particle["color"],
                (int(particle["x"]), int(particle["y"])),
                particle["radius"]
            )
        
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        highlight_radius = max(3, int(self.radius * 0.3))
        highlight_pos = (int(self.x - self.radius * 0.3), int(self.y - self.radius * 0.3))
        pygame.draw.circle(surface, self.config.colors["WHITE"], highlight_pos, highlight_radius)


class Game:
    """Main game class"""
    
    def __init__(self):
        pygame.init()
        self.config = GameConfig()
        
        try:
            cv_build_info = cv2.getBuildInformation()
            has_cuda = "CUDA" in cv_build_info and "YES" in cv_build_info.split("CUDA")[1].split("\n")[0]
            if has_cuda:
                logging.info("OpenCV GPU acceleration available")
                self.config.use_gpu = True
            else:
                logging.info("OpenCV GPU acceleration not available")
                self.config.use_gpu = False
        except Exception as e:
            logging.warning(f"Could not check GPU availability: {e}")
            self.config.use_gpu = False
        
        self.window = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("Catch the Ball")
        self.clock = pygame.time.Clock()
        
        self.resources = ResourceManager(self.config)
        self.hand_detector = HandDetector(self.config)
        
        self.pot = None
        self.balls = []
        self.particles = []
        self.score = 0
        self.high_score = 0
        self.level = 1
        self.game_over = False
        self.paused = False
        self.spawn_timer = 0
        self.game_time = 0
        self.start_time = time.time()
        self.fps_values = []
        
        self.current_state = "menu"
        
        self.reset_game()
        logging.info("Game initialized")
        
    def reset_game(self):
        self.pot = Pot(self.config)
        self.balls = []
        self.particles = []
        self.score = 0
        self.level = 1
        self.game_over = False
        self.paused = False
        self.spawn_timer = pygame.time.get_ticks()
        self.start_time = time.time()
        self.current_state = "game"
        logging.info("Game reset")
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.current_state == "game":
                        self.current_state = "paused"
                    elif self.current_state == "paused":
                        self.current_state = "game"
                elif event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_r and (self.game_over or self.current_state == "game_over"):
                    self.reset_game()
                elif event.key == pygame.K_c:
                    self.config.toggle_control_mode()
                elif event.key == pygame.K_1:
                    self.config.update_difficulty("easy")
                elif event.key == pygame.K_2:
                    self.config.update_difficulty("normal")
                elif event.key == pygame.K_3:
                    self.config.update_difficulty("hard")
                elif event.key == pygame.K_RETURN and self.current_state == "menu":
                    self.current_state = "game"
                    self.reset_game()
                    
        return True
        
    def update_game(self):
        if self.current_state != "game" or self.paused:
            return
            
        self.game_time = time.time() - self.start_time
        
        if self.config.control_mode == "hand":
            hand_data = self.hand_detector.detect_gesture()
            if hand_data and hand_data["hand_x"] is not None:
                self.pot.move_to_position(hand_data["hand_x"])
                
        elif self.config.control_mode == "keyboard":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.pot.move("left")
            if keys[pygame.K_RIGHT]:
                self.pot.move("right")
                
        elif self.config.control_mode == "auto":
            if self.balls:
                closest_ball = min(self.balls, key=lambda ball: ball.y)
                self.pot.move_to_position(closest_ball.x)
        
        self.pot.update()
        
        current_time = pygame.time.get_ticks()
        spawn_interval = self.config.ball_spawn_intervals[self.config.difficulty]
        spawn_interval = max(300, spawn_interval - (self.level * 50))
        
        if current_time - self.spawn_timer >= spawn_interval:
            self.balls.append(Ball(self.config))
            self.spawn_timer = current_time
            
        for ball in self.balls[:]:
            ball.update()
            
            if (self.pot.y <= ball.y + ball.radius <= self.pot.y + self.pot.height and 
                self.pot.x <= ball.x <= self.pot.x + self.pot.width):
                self.score += ball.points
                ball.create_particles(20)
                self.particles.extend(ball.particles)
                ball.particles = []
                self.balls.remove(ball)
                self.level = max(1, self.score // 10 + 1)
                
            elif ball.y - ball.radius > self.config.height:
                self.balls.remove(ball)
                self.game_over = True
                self.current_state = "game_over"
                self.high_score = max(self.high_score, self.score)
                
        for particle in self.particles[:]:
            particle["life"] -= 1
            if particle["life"] <= 0:
                self.particles.remove(particle)
            else:
                particle["x"] += particle["dx"]
                particle["y"] += particle["dy"]
                particle["dy"] += 0.1
                
    def render_game(self):
        fps = self.clock.get_fps()
        self.fps_values.append(fps)
        if len(self.fps_values) > 60:
            self.fps_values.pop(0)
        avg_fps = sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0
        
        self.window.fill(self.config.colors["BLACK"])
        
        for i in range(self.config.height):
            color_val = max(0, 50 - i * 50 // self.config.height)
            pygame.draw.line(
                self.window,
                (0, 0, color_val),
                (0, i),
                (self.config.width, i)
            )
            
        if self.current_state == "menu":
            title = self.resources.render_text("CATCH THE BALL", "large", self.config.colors["YELLOW"])
            start_text = self.resources.render_text("Press ENTER to Start", "medium")
            controls_text = self.resources.render_text("Controls: C to change, 1-3 for difficulty", "small")
            
            self.window.blit(title, [self.config.width // 2 - title.get_width() // 2, 150])
            self.window.blit(start_text, [self.config.width // 2 - start_text.get_width() // 2, 250])
            self.window.blit(controls_text, [self.config.width // 2 - controls_text.get_width() // 2, 300])
            
        elif self.current_state == "game" or self.current_state == "paused":
            self.pot.render(self.window)
            
            for ball in self.balls:
                ball.render(self.window)
                
            for particle in self.particles:
                pygame.draw.circle(
                    self.window,
                    particle["color"],
                    (int(particle["x"]), int(particle["y"])),
                    particle["radius"]
                )
                
            score_text = self.resources.render_text(f"Score: {self.score}", "medium")
            self.window.blit(score_text, [10, 10])
            
            level_text = self.resources.render_text(f"Level: {self.level}", "medium")
            self.window.blit(level_text, [10, 50])
            
            time_text = self.resources.render_text(f"Time: {self.game_time:.1f}s", "medium")
            self.window.blit(time_text, [self.config.width - time_text.get_width() - 10, 10])
            
            control_text = self.resources.render_text(
                f"Control: {self.config.control_mode} (C to change)",
                "small"
            )
            self.window.blit(control_text, [self.config.width - control_text.get_width() - 10, 50])
            
            diff_text = self.resources.render_text(
                f"Difficulty: {self.config.difficulty} (1-3)",
                "small"
            )
            self.window.blit(diff_text, [self.config.width - diff_text.get_width() - 10, 80])
            
            fps_text = self.resources.render_text(f"FPS: {avg_fps:.1f}", "small")
            self.window.blit(fps_text, [10, self.config.height - 30])
            
            if self.current_state == "paused":
                overlay = pygame.Surface((self.config.width, self.config.height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 128))
                self.window.blit(overlay, (0, 0))
                
                pause_text = self.resources.render_text("PAUSED", "large", self.config.colors["WHITE"])
                resume_text = self.resources.render_text("Press ESC to Resume", "medium")
                
                self.window.blit(
                    pause_text,
                    [self.config.width // 2 - pause_text.get_width() // 2, self.config.height // 2 - 50]
                )
                self.window.blit(
                    resume_text,
                    [self.config.width // 2 - resume_text.get_width() // 2, self.config.height // 2 + 20]
                )
                
        elif self.current_state == "game_over":
            overlay = pygame.Surface((self.config.width, self.config.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.window.blit(overlay, (0, 0))
            
            game_over_text = self.resources.render_text("GAME OVER", "large", self.config.colors["RED"])
            score_text = self.resources.render_text(f"Score: {self.score}", "medium")
            high_score_text = self.resources.render_text(f"High Score: {self.high_score}", "medium")
            restart_text = self.resources.render_text("Press R to Restart or Q to Quit", "small")
            
            self.window.blit(
                game_over_text,
                [self.config.width // 2 - game_over_text.get_width() // 2, self.config.height // 2 - 100]
            )
            self.window.blit(
                score_text,
                [self.config.width // 2 - score_text.get_width() // 2, self.config.height // 2 - 20]
            )
            self.window.blit(
                high_score_text,
                [self.config.width // 2 - high_score_text.get_width() // 2, self.config.height // 2 + 20]
            )
            self.window.blit(
                restart_text,
                [self.config.width // 2 - restart_text.get_width() // 2, self.config.height // 2 + 60]
            )
            
        pygame.display.flip()
        
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update_game()
            self.render_game()
            self.clock.tick(self.config.fps)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                
        self.hand_detector.cleanup()
        pygame.quit()
        logging.info("Game terminated")


if __name__ == "__main__":
    try:
        game = Game()
        game.run()
    except Exception as e:
        logging.error(f"Critical error: {e}")
        pygame.quit()
        if 'game' in locals() and game.hand_detector:
            game.hand_detector.cleanup()
        sys.exit(1)
