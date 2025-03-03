import pygame
import random
import cv2
import mediapipe as mp
import logging
import numpy as np
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize Pygame
pygame.init()
logging.info("Pygame initialized")

# Initialize OpenCV webcam with preferred resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    logging.error("Failed to open webcam")
    exit()
logging.info("Webcam initialized")

# Check for GPU support with OpenCV
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    logging.info(f"CUDA-enabled GPU detected: {cv2.cuda.getDevice()}")
    use_gpu = True
else:
    logging.info("No CUDA-enabled GPU detected, using CPU")
    use_gpu = False

# Initialize MediaPipe Hands
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1  # Higher value for better accuracy
    )
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    logging.info("MediaPipe Hands initialized")
except Exception as e:
    logging.error(f"Failed to initialize MediaPipe Hands: {e}")
    cap.release()
    exit()

# Set up display
width = 800
height = 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Catch the Ball")
icon = pygame.Surface((32, 32))
icon.fill((0, 255, 0))
pygame.draw.circle(icon, (255, 0, 0), (16, 16), 10)
pygame.display.set_icon(icon)
logging.info("Display window initialized")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GOLD = (255, 215, 0)
PURPLE = (128, 0, 128)

# Game variables
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)

# Load or create sounds
try:
    catch_sound = pygame.mixer.Sound(pygame.mixer.Sound(bytes(np.sin(np.arange(0, 8000, 100) * 0.5) * 0.5 * 32767, dtype=np.int16)))
    game_over_sound = pygame.mixer.Sound(bytes(np.sin(np.arange(0, 11025, 100) * 0.3) * 0.3 * 32767, dtype=np.int16))
    catch_sound.set_volume(0.3)
    game_over_sound.set_volume(0.3)
    sound_enabled = True
    logging.info("Sounds initialized")
except Exception as e:
    logging.warning(f"Failed to initialize sounds: {e}")
    sound_enabled = False

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.size = random.randint(2, 5)
        self.speed_x = random.uniform(-2, 2)
        self.speed_y = random.uniform(-3, -1)
        self.life = random.randint(10, 30)
    
    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y
        self.life -= 1
        self.speed_y += 0.1  # Gravity effect
        self.size = max(0, self.size - 0.1)
        
    def render(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))

class Pot:
    def __init__(self):
        self.width = 100
        self.height = 20
        self.x = width // 2 - self.width // 2
        self.y = height - 40
        self.speed = 10
        self.target_x = self.x  # For smooth movement
        self.color = GREEN
        self.particles = []

    def move(self, direction):
        if direction == "left":
            self.target_x = max(0, self.target_x - self.speed)
        elif direction == "right":
            self.target_x = min(width - self.width, self.target_x + self.speed)
        
        # Smooth movement
        self.x += (self.target_x - self.x) * 0.3
        
    def move_to(self, position_x):
        # Move directly to a specific position (for mouse or hand tracking)
        self.target_x = max(0, min(width - self.width, position_x - self.width // 2))
    
    def add_particles(self, count=5):
        for _ in range(count):
            self.particles.append(Particle(
                random.randint(int(self.x), int(self.x + self.width)),
                self.y,
                self.color
            ))
    
    def update_particles(self):
        for particle in self.particles[:]:
            particle.update()
            if particle.life <= 0:
                self.particles.remove(particle)

    def render(self, surface):
        # Draw particles behind the pot
        for particle in self.particles:
            particle.render(surface)
        
        # Draw the pot with 3D effect
        pygame.draw.rect(surface, self.color, [self.x, self.y, self.width, self.height])
        pygame.draw.rect(surface, (min(255, self.color[0] + 50), min(255, self.color[1] + 50), min(255, self.color[2] + 50)), 
                        [self.x, self.y, self.width, 5])  # Highlight
        pygame.draw.rect(surface, (max(0, self.color[0] - 50), max(0, self.color[1] - 50), max(0, self.color[2] - 50)), 
                        [self.x, self.y + self.height - 5, self.width, 5])  # Shadow

class Ball:
    def __init__(self, special=False):
        self.radius = 15
        self.x = random.randint(self.radius, width - self.radius)
        self.y = -self.radius
        self.speed = random.uniform(3, 7)  # Variable speed
        self.special = special
        self.color = GOLD if special else RED
        self.points = 5 if special else 1
        self.particles = []
        self.pulse_timer = 0
        self.pulse_dir = 1
    
    def update(self):
        self.y += self.speed
        
        # Pulsing effect for special balls
        if self.special:
            self.pulse_timer += 0.1 * self.pulse_dir
            if self.pulse_timer > 1 or self.pulse_timer < 0:
                self.pulse_dir *= -1
        
        # Trail particles
        if random.random() < 0.2:
            self.particles.append(Particle(self.x, self.y, self.color))
        
        # Update particles
        for particle in self.particles[:]:
            particle.update()
            if particle.life <= 0:
                self.particles.remove(particle)

    def render(self, surface):
        # Draw particles behind the ball
        for particle in self.particles:
            particle.render(surface)
        
        # Draw the ball with glow effect if special
        if self.special:
            pulse_radius = self.radius + 3 * self.pulse_timer
            glow_surface = pygame.Surface((int(pulse_radius*2.5), int(pulse_radius*2.5)), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*self.color[:3], 100), (int(pulse_radius*1.25), int(pulse_radius*1.25)), int(pulse_radius*1.2))
            surface.blit(glow_surface, (self.x - pulse_radius*1.25, self.y - pulse_radius*1.25))
        
        # Main ball
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        # Highlight
        highlight_pos = (int(self.x - self.radius * 0.3), int(self.y - self.radius * 0.3))
        pygame.draw.circle(surface, (min(255, self.color[0] + 70), min(255, self.color[1] + 70), min(255, self.color[2] + 70)), 
                           highlight_pos, int(self.radius * 0.3))

class PowerUp:
    def __init__(self):
        self.radius = 12
        self.x = random.randint(self.radius, width - self.radius)
        self.y = -self.radius
        self.speed = 4
        self.type = random.choice(["wide", "slow", "multi"])
        
        if self.type == "wide":
            self.color = BLUE
        elif self.type == "slow":
            self.color = PURPLE
        else:
            self.color = GREEN
        
        self.active = True
        self.pulse_timer = 0
        self.pulse_dir = 1
    
    def update(self):
        self.y += self.speed
        
        # Pulsing effect
        self.pulse_timer += 0.1 * self.pulse_dir
        if self.pulse_timer > 1 or self.pulse_timer < 0:
            self.pulse_dir *= -1
    
    def render(self, surface):
        pulse_radius = self.radius + 2 * self.pulse_timer
        
        # Draw outer glow
        glow_surface = pygame.Surface((int(pulse_radius*2.5), int(pulse_radius*2.5)), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.color[:3], 100), (int(pulse_radius*1.25), int(pulse_radius*1.25)), int(pulse_radius*1.2))
        surface.blit(glow_surface, (self.x - pulse_radius*1.25, self.y - pulse_radius*1.25))
        
        # Draw power-up
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw icon inside
        if self.type == "wide":
            pygame.draw.rect(surface, WHITE, [self.x - 5, self.y - 2, 10, 4])
        elif self.type == "slow":
            pygame.draw.line(surface, WHITE, [self.x - 3, self.y - 3], [self.x + 3, self.y + 3], 2)
            pygame.draw.line(surface, WHITE, [self.x + 3, self.y - 3], [self.x - 3, self.y + 3], 2)
        else:  # multi
            pygame.draw.circle(surface, WHITE, (int(self.x) - 3, int(self.y)), 2)
            pygame.draw.circle(surface, WHITE, (int(self.x) + 3, int(self.y)), 2)

def detect_hand_gesture():
    try:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame from webcam")
            return None, None
        
        frame = cv2.flip(frame, 1)
        
        # Use GPU if available
        if use_gpu:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_rgb_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
            rgb_frame = gpu_rgb_frame.download()
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        direction = None
        hand_x = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks or [], results.multi_handedness or []):
                # Draw landmarks with improved styling
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                
                # Extract hand position for direct control
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                
                # Get screen coordinates from normalized coordinates
                hand_x = int(index_finger_tip.x * frame.shape[1])
                hand_y = int(index_finger_tip.y * frame.shape[0])
                
                # Map webcam coordinates to game window coordinates
                hand_x = int(hand_x * (width / frame.shape[1]))
                
                # For basic left/right movement
                if wrist.x < 0.3:
                    direction = "left"
                elif wrist.x > 0.7:
                    direction = "right"
                
                # Display hand information
                label = handedness.classification[0].label
                wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])
                cv2.putText(frame, f"{label} hand", (wrist_x - 30, wrist_y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw control point
                cv2.circle(frame, (hand_x, hand_y), 8, (0, 255, 0), -1)
        
        # Enhanced visualization
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(frame, "Hand Detection Feed", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Hand Detection Feed", frame)
        cv2.moveWindow("Hand Detection Feed", width + 10, 0)
        return direction, hand_x
    
    except Exception as e:
        logging.error(f"Error in detect_hand_gesture: {e}")
        return None, None

def create_explosion(x, y, color, count=15):
    particles = []
    for _ in range(count):
        particles.append(Particle(x, y, color))
    return particles

def draw_background(surface):
    # Simple gradient background
    for y in range(0, height, 2):
        color_value = int(y / height * 100)
        pygame.draw.line(surface, (color_value, color_value, color_value + 20), 
                         (0, y), (width, y))
    
    # Stars
    for _ in range(20):
        x = random.randint(0, width)
        y = random.randint(0, height // 2)
        size = random.randint(1, 3)
        brightness = random.randint(150, 255)
        pygame.draw.circle(surface, (brightness, brightness, brightness), (x, y), size)

def main():
    pot = Pot()
    balls = []
    power_ups = []
    particles = []
    score = 0
    high_score = 0
    level = 1
    lives = 3
    game_over = False
    paused = False
    
    # Game state variables
    spawn_timer = pygame.time.get_ticks()
    spawn_interval = 2000  # Spawn a new ball every 2 seconds
    power_up_timer = pygame.time.get_ticks()
    power_up_interval = 10000  # Spawn power-up every 10 seconds
    
    # Power-up effects
    pot_width_timer = 0
    slow_motion_timer = 0
    multi_ball_timer = 0
    
    # Performance monitoring
    frame_times = []
    last_frame_time = time.time()
    
    # Game state
    game_state = "menu"  # "menu", "playing", "game_over"
    
    logging.info("Game loop starting")
    
    while True:
        # Calculate frame time and FPS
        current_time = time.time()
        dt = current_time - last_frame_time
        last_frame_time = current_time
        frame_times.append(dt)
        if len(frame_times) > 60:
            frame_times.pop(0)
        average_fps = 1.0 / (sum(frame_times) / len(frame_times))
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    return
                elif event.key == pygame.K_r and game_over:
                    pot = Pot()
                    balls = []
                    power_ups = []
                    particles = []
                    score = 0
                    level = 1
                    lives = 3
                    game_over = False
                    spawn_timer = pygame.time.get_ticks()
                    power_up_timer = pygame.time.get_ticks()
                    logging.info("Game restarted")
                elif event.key == pygame.K_p:
                    paused = not paused
                    logging.info(f"Game {'paused' if paused else 'resumed'}")
                elif event.key == pygame.K_RETURN:
                    if game_state == "menu":
                        game_state = "playing"
                    elif game_state == "game_over":
                        game_state = "menu"
                        game_over = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game_state == "menu":
                    game_state = "playing"
                elif game_state == "game_over":
                    game_state = "menu"
                    game_over = False
        
        # Main menu
        if game_state == "menu":
            window.fill(BLACK)
            draw_background(window)
            
            title_font = pygame.font.SysFont(None, 72)
            title_text = title_font.render("Catch the Ball", True, WHITE)
            window.blit(title_text, [width // 2 - title_text.get_width() // 2, 100])
            
            instructions_font = pygame.font.SysFont(None, 28)
            instructions = [
                "Move your hand left/right to control the pot",
                "Catch falling balls to score points",
                "Special golden balls are worth 5 points",
                "Collect power-ups for special abilities",
                "",
                "Press ENTER or click to start"
            ]
            
            for i, line in enumerate(instructions):
                text = instructions_font.render(line, True, WHITE)
                window.blit(text, [width // 2 - text.get_width() // 2, 200 + i * 40])
            
            # Draw sample pot and ball
            sample_pot = Pot()
            sample_pot.x = width // 2 - sample_pot.width // 2
            sample_pot.y = 450
            sample_pot.render(window)
            
            sample_ball = Ball()
            sample_ball.x = width // 2
            sample_ball.y = 420
            sample_ball.render(window)
            
            # Animate
            if random.random() < 0.1:
                pot.add_particles(2)
            pot.update_particles()
            
            pygame.display.update()
            clock.tick(60)
            
            # Process hand tracking in the background
            detect_hand_gesture()
            
            continue
        
        # Game over screen
        if game_state == "game_over":
            window.fill(BLACK)
            draw_background(window)
            
            game_over_font = pygame.font.SysFont(None, 72)
            game_over_text = game_over_font.render("Game Over!", True, RED)
            window.blit(game_over_text, [width // 2 - game_over_text.get_width() // 2, 150])
            
            score_font = pygame.font.SysFont(None, 48)
            final_score_text = score_font.render(f"Final Score: {score}", True, WHITE)
            window.blit(final_score_text, [width // 2 - final_score_text.get_width() // 2, 220])
            
            high_score_text = score_font.render(f"High Score: {high_score}", True, GOLD)
            window.blit(high_score_text, [width // 2 - high_score_text.get_width() // 2, 270])
            
            restart_font = pygame.font.SysFont(None, 36)
            restart_text = restart_font.render("Press ENTER or click to continue", True, WHITE)
            window.blit(restart_text, [width // 2 - restart_text.get_width() // 2, 350])
            
            pygame.display.update()
            clock.tick(60)
            
            # Process hand tracking in the background
            detect_hand_gesture()
            
            continue
        
        # Don't update game logic if paused
        if paused:
            # Render pause screen
            pause_overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            pause_overlay.fill((0, 0, 0, 128))
            window.blit(pause_overlay, (0, 0))
            
            pause_font = pygame.font.SysFont(None, 72)
            pause_text = pause_font.render("PAUSED", True, WHITE)
            window.blit(pause_text, [width // 2 - pause_text.get_width() // 2, height // 2 - 36])
            
            pause_hint = font.render("Press P to Resume", True, WHITE)
            window.blit(pause_hint, [width // 2 - pause_hint.get_width() // 2, height // 2 + 36])
            
            pygame.display.update()
            clock.tick(60)
            
            # Process hand tracking in the background
            detect_hand_gesture()
            
            continue
        
        if not game_over:
            # Detect hand gesture and move pot
            direction, hand_x = detect_hand_gesture()
            
            if hand_x is not None:
                # Direct control with hand position
                pot.move_to(hand_x)
            elif direction:
                # Fallback to left/right movement
                pot.move(direction)
            
            # Update power-up timers
            if pot_width_timer > 0:
                pot_width_timer -= 1
                pot.width = 150  # Wider pot
                pot.color = BLUE
                if pot_width_timer == 0:
                    pot.width = 100  # Reset to normal
                    pot.color = GREEN
            
            if slow_motion_timer > 0:
                slow_motion_timer -= 1
                # Ball speed already adjusted in the ball update
            
            if multi_ball_timer > 0:
                multi_ball_timer -= 1
            
            # Spawn new balls
            current_time = pygame.time.get_ticks()
            if current_time - spawn_timer >= spawn_interval:
                # Chance for special balls increases with level
                special_chance = min(0.05 * level, 0.3)
                is_special = random.random() < special_chance
                
                balls.append(Ball(special=is_special))
                
                # Spawn multiple balls if multi-ball power-up is active
                if multi_ball_timer > 0 and random.random() < 0.5:
                    balls.append(Ball(special=random.random() < special_chance))
                
                spawn_timer = current_time
                # Adjust spawn interval based on level
                spawn_interval = max(500, 2000 - level * 100)
            
            # Spawn power-ups
            if current_time - power_up_timer >= power_up_interval:
                power_ups.append(PowerUp())
                power_up_timer = current_time
            
            # Update balls with slow motion effect
            slow_factor = 0.5 if slow_motion_timer > 0 else 1.0
            for ball in balls[:]:
                # Apply slow motion if active
                original_speed = ball.speed
                if slow_motion_timer > 0:
                    ball.speed *= slow_factor
                
                ball.update()
                
                # Restore original speed
                ball.speed = original_speed
                
                # Check collision with pot
                if (pot.y <= ball.y + ball.radius <= pot.y + pot.height and 
                    pot.x <= ball.x <= pot.x + pot.width):
                    
                    # Create explosion effect
                    particles.extend(create_explosion(ball.x, ball.y, ball.color))
                    pot.add_particles(3)
                    
                    # Add score
                    score += ball.points
                    
                    # Level up
                    if score >= level * 20:
                        level += 1
                        pot.add_particles(10)  # Celebrate level up
                    
                    balls.remove(ball)
                    
                    # Play sound
                    if sound_enabled:
                        catch_sound.play()
                
                # Remove balls that go off screen
                elif ball.y > height:
                    balls.remove(ball)
                    
                    # Lose a life
                    lives -= 1
                    particles.extend(create_explosion(ball.x, height - 5, (255, 0, 0), 20))
                    
                    if lives <= 0:
                        game_over = True
                        game_state = "game_over"
                        high_score = max(high_score, score)
                        
                        # Play game over sound
                        if sound_enabled:
                            game_over_sound.play()
            
            # Update power-ups
            for power_up in power_ups[:]:
                power_up.update()
                
                # Check collision with pot
                if (pot.y <= power_up.y + power_up.radius <= pot.y + pot.height and 
                    pot.x <= power_up.x <= pot.x + power_up.width):
                    
                    # Apply power-up effect
                    if power_up.type == "wide":
                        pot_width_timer = 300  # 5 seconds at 60 FPS
                        pot.color = BLUE
                    elif power_up.type == "slow":
                        slow_motion_timer = 300
                    else:  # multi
                        multi_ball_timer = 300
                    
                    # Create effect
                    particles.extend(create_explosion(power_up.x, power_up.y, power_up.color, 20))
                    
                    power_ups.remove(power_up)
                
                # Remove power-ups that go off screen
                elif power_up.y > height:
                    power_ups.remove(power_up)
            
            # Update particles
            for particle in particles[:]:
                particle.update()
                if particle.life <= 0:
                    particles.remove(particle)
            
            pot.update_particles()
        
        # Render game
        window.fill(BLACK)
        draw_background(window)
        
        # Draw particles behind other elements
        for particle in particles:
            particle.render(window)
        
        # Draw pot and balls
        pot.render(window)
        
        for ball in balls:
            ball.render(window)
        
        for power_up in power_ups:
            power_up.render(window)
        
        # Draw active power-up indicators
        power_up_y = 50
        if pot_width_timer > 0:
            wide_text = small_font.render(f"Wide Pot: {pot_width_timer // 60}s", True, BLUE)
            window.blit(wide_text, [10, power_up_y])
            power_up_y += 25
        
        if slow_motion_timer > 0:
            slow_text = small_font.render(f"Slow Motion: {slow_motion_timer // 60}s", True, PURPLE)
            window.blit(slow_text, [10, power_up_y])
            power_up_y += 25
        
        if multi_ball_timer > 0:
            multi_text = small_font.render(f"Multi Ball: {multi_ball_timer // 60}s", True, GREEN)
            window.blit(multi_text, [10, power_up_y])
        
        # Display score and lives
        score_text = font.render(f"Score: {score}", True, WHITE)
        level_text = font.render(f"Level: {level}", True, WHITE)
        lives_text = font.render(f"Lives: {lives}", True, RED)
        fps_text = small_font.render(f"FPS: {int(average_fps)}", True, WHITE)
        
        window.blit(score_text, [10, 10])
        window.blit(level_text, [width - level_text.get_width() - 10, 10])
        window.blit(lives_text, [width // 2 - lives_text.get_width() // 2, 10])
        window.blit(fps_text, [width - fps_text.get_width() - 10, height - 20])
        
        if game_over:
            game_over_text = font.render("Game Over!", True, RED)
            restart_text = font.render("Press R to Restart or Q to Quit", True, WHITE)
            window.blit(game_over_text, [width // 2 - game_over_text.get_width() // 2, height // 2 - 20])
            window.blit(restart_text, [width // 2 - restart_text.get_width() // 2, height // 2 + 20])
        
        pygame.display.update()
        clock.tick(60)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Q key pressed via OpenCV window")
            break
    
    cap.release()
    cv2.destroyAllWindows()