import pygame
import moviepy.editor as mpy

def pygame_view(frames, scores, output_video_path="tetris_game_test.mp4"):
    pygame.init()

    ROWS, COLS = frames[0].shape # Grid dimensions
    CELL_SIZE = 40  # Size of each block in pixels
    WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE + 40  # Extra height for the score display

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLOCK_COLOR = (0, 128, 255)  # Color for the blocks
    TEXT_COLOR = (255, 255, 255)

    # Create Pygame window
    window = pygame.display.set_mode((WIDTH, HEIGHT))  # Main display window
    pygame.display.set_caption("Tetris")
    screen = pygame.Surface((WIDTH, HEIGHT))  # Off-screen surface for video capture
    clock = pygame.time.Clock()  # For smooth updates

    # Font for displaying the score
    font = pygame.font.Font(None, 36)

    # List to store frames for the video
    video_frames = []

    running = True
    for frame_index, frame in enumerate(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        # Fill screen with background color
        screen.fill(BLACK)

        # Draw the grid
        for row in range(ROWS):
            for col in range(COLS):
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if frame[row][col]:
                    pygame.draw.rect(screen, BLOCK_COLOR, rect)
                pygame.draw.rect(screen, WHITE, rect, 1)  # Grid lines

        # Draw the score at the bottom
        score_text = font.render(f"Score: {scores[frame_index]}", True, TEXT_COLOR)
        screen.blit(score_text, (10, ROWS * CELL_SIZE + 5))

        # Copy the rendered frame to the display window
        window.blit(screen, (0, 0))
        pygame.display.flip()

        # Capture the frame for video
        frame_data = pygame.surfarray.array3d(screen)
        video_frames.append(frame_data.swapaxes(0, 1))  # Swap axes to match moviepy's format

        clock.tick(2)  # 2 frames per second

    # Save the video using moviepy
    clip = mpy.ImageSequenceClip(video_frames, fps=2)
    clip.write_videofile(output_video_path, codec="libx264")

    pygame.quit()