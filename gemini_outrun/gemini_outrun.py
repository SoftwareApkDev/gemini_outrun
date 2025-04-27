import pygame
import random
import os
import google.generativeai as genai
import threading # Used for non-blocking AI calls (optional but recommended)
import time # For timing AI calls

from dotenv import load_dotenv

# --- Game Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (80, 80, 80)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
SKY_COLOR = (100, 150, 255)
ROAD_LINE_COLOR = (200, 200, 200)

# Road Properties (Conceptual)
# These control the perspective effect. screen_y -> depth -> width/offset
HORIZON_Y = 100 # Y coordinate of the horizon line
ROAD_START_Y = SCREEN_HEIGHT # Y coordinate where road starts at bottom
ROAD_WIDTH_AT_START = 500 # Pixels wide at the bottom
ROAD_WIDTH_AT_HORIZON = 10 # Pixels wide at the horizon
ROAD_SEGMENT_HEIGHT = 5 # Height of each horizontal road segment for drawing

# Player Car Properties
PLAYER_CAR_WIDTH = 50
PLAYER_CAR_HEIGHT = 30
PLAYER_X_START = SCREEN_WIDTH // 2
PLAYER_Y_FIXED = SCREEN_HEIGHT - PLAYER_CAR_HEIGHT - 10 # Player car stays at a fixed Y

# Speed and Movement
ACCELERATION = 0.5 # How fast speed increases per frame
DECELERATION = 1.0 # How fast speed decreases per frame when braking/no gas
FRICTION = 0.1 # Constant speed loss per frame
MAX_SPEED = 750 # Max speed in internal units
STEERING_SPEED = 5 # How fast player moves left/right
PLAYER_MAX_SIDE_MOVEMENT = 150 # Limit how far player can move horizontally relative to screen center

# Speed Conversion (Internal units to km/h)
# This is a arbitrary scaling factor. Adjust to feel right.
SPEED_TO_KMH = 1.5

# Opponent Properties
OPPONENT_COUNT = 5
OPPONENT_Z_MIN = 50 # Minimum starting distance (depth)
OPPONENT_Z_MAX = 500 # Maximum starting distance (depth)
OPPONENT_SPEED_MIN = 0.9 # Opponent speed relative to player (0.9 = 90% of player speed)
OPPONENT_SPEED_MAX = 1.05 # Can be slightly faster than player

# Gemini AI Integration
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
GEMINI_PROMPT_INTERVAL_SECONDS = 20 # How often to call the AI
GEMINI_DISPLAY_DURATION_SECONDS = 10 # How long to display AI text
GEMINI_INITIAL_DELAY_SECONDS = 5 # Delay before first AI call

# --- Global Variables ---
current_speed = 0 # Internal speed units
player_x_offset = 0 # Horizontal offset from screen center for player movement
road_curve = 0 # Simple curve factor (not implemented in drawing yet)
road_z_offset = 0 # Overall track progression (conceptual depth offset)

opponents = [] # List to hold opponent data

gemini_model = None
gemini_text = ""
gemini_text_display_end_time = 0
last_gemini_prompt_time = 0
gemini_thread = None # To hold the AI request thread

# --- Helper Functions ---

def draw_text(surface, text, size, color, x, y):
    """Helper to draw text on the screen."""
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    surface.blit(text_surface, text_rect)

def perspective_scale(y, base_width, base_y, horizon_y):
    """Calculate perspective scale based on screen Y position."""
    # Linear interpolation for simplicity
    # scale goes from 1 at base_y to a small value at horizon_y
    if y >= base_y:
        return 1.0
    if y <= horizon_y:
        return 0.0 # Or some minimum scale
    # Inverse linear interpolation on distance from horizon
    distance_from_horizon = y - horizon_y
    total_distance = base_y - horizon_y
    t = distance_from_horizon / total_distance # 0 at horizon, 1 at base_y
    # Width scales linearly with t (simple)
    # A more realistic perspective uses 1/distance, but linear is easier here
    # scale = t
    # Or scale width more directly:
    width_diff = ROAD_WIDTH_AT_START - ROAD_WIDTH_AT_HORIZON
    current_width = ROAD_WIDTH_AT_HORIZON + width_diff * t
    scale_factor = current_width / ROAD_WIDTH_AT_START # Scale relative to base width
    return scale_factor

def get_road_params_at_y(y):
    """Calculate width and center x for the road at a given screen Y."""
    if y < HORIZON_Y:
        return 0, SCREEN_WIDTH // 2
    if y > ROAD_START_Y:
        y = ROAD_START_Y # Clamp
    if y < HORIZON_Y:
         y = HORIZON_Y # Clamp

    # Distance from the horizon (conceptual depth)
    distance_from_horizon = y - HORIZON_Y
    total_distance = ROAD_START_Y - HORIZON_Y

    # Simple linear scaling for width and horizontal shift
    t = distance_from_horizon / total_distance # 0 at horizon, 1 at base_y

    road_width = ROAD_WIDTH_AT_HORIZON + (ROAD_WIDTH_AT_START - ROAD_WIDTH_AT_HORIZON) * t
    # Player horizontal movement shifts the road center.
    # The shift amount scales with depth (less shift far away, more shift close up)
    road_center_x = SCREEN_WIDTH // 2 + player_x_offset * t * (ROAD_WIDTH_AT_START / SCREEN_WIDTH) # Scale player offset by t and a factor

    return road_width, road_center_x

# --- Game Objects (Simplified) ---

class Opponent(pygame.sprite.Sprite):
    def __init__(self, image, z, x):
        super().__init__()
        self.original_image = image
        self.image = image # Scaled version
        self.rect = self.image.get_rect()
        self.z = z # World depth position
        self.x = x # World horizontal position (-1 to 1 relative to road center)
        self.speed_factor = random.uniform(OPPONENT_SPEED_MIN, OPPONENT_SPEED_MAX) # Speed relative to player

    def update(self, dt, player_speed):
        global road_z_offset
        # Move the opponent based on player speed and its own speed factor
        # Z position decreases as road scrolls under car
        self.z -= player_speed * self.speed_factor * dt * 0.1 # Adjust factor for feel

        # Simple horizontal movement (wiggle)
        # self.x += math.sin(time.time() * 2 + self.z) * 0.005 # Add some basic AI wiggle

        # Check if opponent is behind player (z < 0) - handled outside

    def draw(self, surface):
        # Calculate screen position based on Z using road perspective
        # Need to find the screen Y corresponding to this Z
        # This is the inverse of get_road_params_at_y's depth calculation

        # Simple inverse: Assuming linear depth mapping
        # distance_from_horizon = self.z * depth_to_y_scale
        # screen_y = HORIZON_Y + distance_from_horizon
        # This is tricky with our simple y-based road.
        # Let's iterate y to find approximate z, or use a formula based on our get_road_params_at_y
        # Approx Z = (y - HORIZON_Y) / depth_scale. So y = HORIZON_Y + Z * depth_scale
        # What is depth_scale? From get_road_params_at_y, depth is proportional to (y - HORIZON_Y).
        # Let's try mapping Z directly to screen Y based on min/max Z and min/max Y
        total_road_y_range = ROAD_START_Y - HORIZON_Y
        total_road_z_range = OPPONENT_Z_MAX # Assume 0 is near, OPPONENT_Z_MAX is far limit

        # Map Z to a value between 0 (near) and 1 (far) relative to our conceptual road range
        # Clamp Z to stay within visible range for perspective calculation
        clamped_z = max(0.1, min(OPPONENT_Z_MAX * 1.2, self.z)) # Ensure Z is positive and within reasonable bound

        # Simple inverse linear mapping of Z to screen Y (needs tuning)
        # A more accurate approach involves mapping Z to the perspective factor 't' and then to y
        # t = clamped_z / OPPONENT_Z_MAX # Simple Z -> t (0 at near, 1 at far) - this is wrong way
        # t should be low for high z (far) and high for low z (near)
        # t = 1.0 - (clamped_z / (OPPONENT_Z_MAX + 100)) # Map Z inversely to t (clamped)
        # screen_y = HORIZON_Y + total_road_y_range * t # This mapping is also simplistic

        # Let's use a simplified perspective mapping based on a base distance
        # Closer objects are drawn lower and larger. Z is distance *away*
        # screen_y = HORIZON_Y + perspective_mapping_func(clamped_z)
        # screen_scale = scale_mapping_func(clamped_z)

        # A common method: y = const / Z (+ offset), scale = const / Z
        PERSPECTIVE_DEPTH_FACTOR = 400 # Adjust for feel

        screen_y = HORIZON_Y + PERSPECTIVE_DEPTH_FACTOR / clamped_z
        # Clamp screen_y to be within visible road area
        screen_y = max(HORIZON_Y + ROAD_SEGMENT_HEIGHT, min(ROAD_START_Y - self.rect.height / 2, screen_y))


        # Calculate width and horizontal center at this screen_y using road logic
        road_width_at_y, road_center_at_y = get_road_params_at_y(screen_y)

        # Calculate scale based on perspective at this Y
        # scale = perspective_scale(screen_y, ROAD_WIDTH_AT_START, ROAD_START_Y, HORIZON_Y)
        # Or directly from Z
        scale = PERSPECTIVE_DEPTH_FACTOR / (clamped_z * 5) # Adjust factor 5 for object size relative to road

        if scale < 0.1: scale = 0.1 # Minimum scale
        if scale > 2.0: scale = 2.0 # Maximum scale (shouldn't happen if Z is clamped)


        # Scale the opponent image
        scaled_width = int(self.original_image.get_width() * scale)
        scaled_height = int(self.original_image.get_height() * scale)

        # Prevent scaling to zero or negative size
        if scaled_width <= 0 or scaled_height <= 0:
            return

        self.image = pygame.transform.scale(self.original_image, (scaled_width, scaled_height))
        self.rect = self.image.get_rect()

        # Calculate screen X position
        # Opponent X is relative to the road center at its depth
        # The road center itself is shifted by player_x_offset scaled by perspective
        # Opponent's world_x needs to be mapped to screen_x based on perspective
        # screen_x = road_center_at_y + self.x * (road_width_at_y / 2) # Simple mapping to road width
        # Or scale world_x directly
        screen_x = road_center_at_y + self.x * scaled_width # Map x relative to object width scale

        self.rect.center = (int(screen_x), int(screen_y))

        surface.blit(self.image, self.rect)


# --- Road Drawing Function ---
def draw_road(surface):
    # Draw road segments from bottom up to the horizon
    segment_y = ROAD_START_Y
    row_index = 0 # To alternate colors

    while segment_y > HORIZON_Y:
        road_width, road_center_x = get_road_params_at_y(segment_y)

        left = int(road_center_x - road_width // 2)
        right = int(road_center_x + road_width // 2)
        top = int(segment_y - ROAD_SEGMENT_HEIGHT)
        bottom = int(segment_y)

        # Clamp drawing to screen bounds
        left = max(0, left)
        right = min(SCREEN_WIDTH, right)
        top = max(HORIZON_Y, top)
        bottom = min(ROAD_START_Y, bottom)

        # Determine color based on row index
        color = GREEN if (row_index // 10) % 2 == 0 else DARK_GREEN # Wide stripes

        # Draw the road segment
        pygame.draw.rect(surface, color, (left, top, right - left, bottom - top))

        # Draw road lines (simplified - just center line)
        # Line width scales with road width
        line_width = max(1, int(road_width * 0.02)) # 2% of road width
        center_line_x = road_center_x
        pygame.draw.line(surface, ROAD_LINE_COLOR, (center_line_x, top), (center_line_x, bottom), line_width)


        segment_y -= ROAD_SEGMENT_HEIGHT
        row_index += 1


# --- Gemini AI Function ---
def get_gemini_commentary_async(prompt):
    """Fetches commentary from Gemini in a separate thread."""
    global gemini_text, gemini_text_display_end_time, gemini_model
    try:
        if gemini_model is None:
             gemini_text = "AI: Initializing..."
             return

        # The model.generate_content call is blocking
        response = gemini_model.generate_content(prompt)
        # Simple error check
        if hasattr(response, 'text'):
            gemini_text = "AI: " + response.text.strip()
            gemini_text_display_end_time = time.time() + GEMINI_DISPLAY_DURATION_SECONDS
        else:
             # Handle potential blocked content or errors
             gemini_text = "AI: (Commentary blocked or error)"
             print("Gemini response blocked or error:", response)

    except Exception as e:
        gemini_text = f"AI Error: {e}"
        print(f"Gemini API Error: {e}")
        # Still display the error for a bit
        gemini_text_display_end_time = time.time() + 5


def start_gemini_commentary_thread(prompt):
    """Starts the Gemini commentary thread if one isn't already running."""
    global gemini_thread
    if gemini_thread is None or not gemini_thread.is_alive():
        gemini_thread = threading.Thread(target=get_gemini_commentary_async, args=(prompt,))
        gemini_thread.start()
        # Update display immediately to show "Requesting..." or similar
        global gemini_text, gemini_text_display_end_time
        gemini_text = "AI: Requesting commentary..."
        gemini_text_display_end_time = time.time() + 5 # Display this message briefly


# --- Main Game Function ---
def main():
    global current_speed, player_x_offset, road_z_offset, opponents
    global gemini_model, last_gemini_prompt_time, gemini_text, gemini_text_display_end_time

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Python Outrun Flash")
    clock = pygame.time.Clock()

    # Load assets (placeholders)
    try:
        player_car_image = pygame.Surface([PLAYER_CAR_WIDTH, PLAYER_CAR_HEIGHT])
        player_car_image.fill(WHITE) # Simple white rectangle
        pygame.draw.rect(player_car_image, BLACK, (0, 0, PLAYER_CAR_WIDTH, PLAYER_CAR_HEIGHT), 2) # Outline
        pygame.draw.circle(player_car_image, BLACK, (PLAYER_CAR_WIDTH//4, PLAYER_CAR_HEIGHT), 5) # Wheels
        pygame.draw.circle(player_car_image, BLACK, (PLAYER_CAR_WIDTH*3//4, PLAYER_CAR_HEIGHT), 5)

        opponent_car_image = pygame.Surface([PLAYER_CAR_WIDTH * 0.8, PLAYER_CAR_HEIGHT * 0.8]) # Smaller opponents
        opponent_car_image.fill((255, 0, 0)) # Red rectangle
        pygame.draw.rect(opponent_car_image, BLACK, (0, 0, PLAYER_CAR_WIDTH*0.8, PLAYER_CAR_HEIGHT*0.8), 2)

    except pygame.error as e:
        print(f"Could not load assets: {e}")
        player_car_image = pygame.Surface([PLAYER_CAR_WIDTH, PLAYER_CAR_HEIGHT])
        player_car_image.fill(WHITE)
        opponent_car_image = pygame.Surface([PLAYER_CAR_WIDTH * 0.8, PLAYER_CAR_HEIGHT * 0.8])
        opponent_car_image.fill((255, 0, 0))

    try:
        load_dotenv()
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        # Optional: Test initial connection/prompt
        # print("Gemini model loaded. Testing...")
        # test_response = gemini_model.generate_content("Hello, testing connection.")
        # print("Test response:", test_response.text)
        last_gemini_prompt_time = time.time() - GEMINI_PROMPT_INTERVAL_SECONDS + GEMINI_INITIAL_DELAY_SECONDS # Trigger first prompt soon
    except Exception as e:
        print(f"Failed to initialize Gemini API: {e}")
        gemini_model = None
        gemini_text = f"AI: Initialization failed: {e}"
        gemini_text_display_end_time = time.time() + 10


    # Create initial opponents
    for _ in range(OPPONENT_COUNT):
        z = random.uniform(OPPONENT_Z_MIN, OPPONENT_Z_MAX)
        x = random.uniform(-0.8, 0.8) # Random horizontal position within road
        opponents.append(Opponent(opponent_car_image, z, x))

    # --- Game Loop ---
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0 # Delta time in seconds

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Add keyboard events for acceleration/braking if desired
            # For now, speed increases with up arrow, decreases with down

        # --- Get Player Input (Continuous) ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player_x_offset -= STEERING_SPEED
        if keys[pygame.K_RIGHT]:
            player_x_offset += STEERING_SPEED

        if keys[pygame.K_UP]:
            current_speed += ACCELERATION
        elif keys[pygame.K_DOWN]:
            current_speed -= DECELERATION
        else:
            current_speed -= FRICTION # Apply friction when not accelerating/braking

        # Clamp speed
        current_speed = max(0, min(MAX_SPEED, current_speed))

        # Clamp player horizontal offset
        player_x_offset = max(-PLAYER_MAX_SIDE_MOVEMENT, min(PLAYER_MAX_SIDE_MOVEMENT, player_x_offset))


        # --- Update Game State ---

        # Move the road/track progression based on speed
        road_z_offset += current_speed * dt * 5 # Adjust factor for how fast road scrolls

        # Update opponents
        opponents_to_remove = []
        for opponent in opponents:
            opponent.update(dt, current_speed)
            # Check if opponent is behind the player plane (z < 0)
            if opponent.z < 0:
                opponents_to_remove.append(opponent)

        # Remove off-screen opponents
        for opponent in opponents_to_remove:
            opponents.remove(opponent)
            # Add a new opponent further down the track
            new_z = random.uniform(OPPONENT_Z_MAX * 0.8, OPPONENT_Z_MAX * 1.5) # Ensure new one is far away
            new_x = random.uniform(-0.8, 0.8)
            opponents.append(Opponent(opponent_car_image, new_z, new_x))


        # --- Gemini AI Update ---
        current_time = time.time()
        if gemini_model and current_time - last_gemini_prompt_time > GEMINI_PROMPT_INTERVAL_SECONDS:
            # Construct a simple prompt based on game state
            speed_kmh = int(current_speed * SPEED_TO_KMH)
            race_prompt = f"You are a race commentator. Give a short (1-2 sentences) exciting comment about the current race situation. Player speed is {speed_kmh} km/h."
            if len(opponents) > 0:
                # Find nearest opponent to mention
                nearest_opponent = min(opponents, key=lambda opp: opp.z)
                if nearest_opponent.z < 100: # Only mention if relatively close
                     race_prompt += f" An opponent is nearby (distance: {int(nearest_opponent.z)})."
            race_prompt += " Keep it dynamic and brief."

            start_gemini_commentary_thread(race_prompt)
            last_gemini_prompt_time = current_time # Reset timer immediately

        # Clear expired AI text
        if current_time > gemini_text_display_end_time:
            gemini_text = ""


        # --- Drawing ---

        # Draw sky
        screen.fill(SKY_COLOR)

        # Draw road
        draw_road(screen)

        # Draw opponents (sort by Z to draw further ones first)
        opponents.sort(key=lambda opp: opp.z, reverse=True) # Draw far to near
        for opponent in opponents:
             opponent.draw(screen)


        # Draw player car
        player_car_rect = player_car_image.get_rect(center=(SCREEN_WIDTH // 2, PLAYER_Y_FIXED))
        screen.blit(player_car_image, player_car_rect)

        # Draw UI (Speed)
        speed_kmh = int(current_speed * SPEED_TO_KMH)
        draw_text(screen, f"Speed: {speed_kmh} km/h", 30, WHITE, 100, 30)

        # Draw Gemini Text
        if gemini_text:
             # Draw a semi-transparent background for the text
            text_bg_rect = pygame.Rect(50, 50, SCREEN_WIDTH - 100, 40)
            s = pygame.Surface((text_bg_rect.width, text_bg_rect.height), pygame.SRCALPHA)
            s.fill((0, 0, 0, 128)) # Black with 128 alpha (half transparent)
            screen.blit(s, text_bg_rect)
            draw_text(screen, gemini_text, 25, WHITE, SCREEN_WIDTH // 2, 70)


        # --- Update Display ---
        pygame.display.flip()

    # --- Game End ---
    pygame.quit()

# --- Run the Game ---
if __name__ == "__main__":
    main()