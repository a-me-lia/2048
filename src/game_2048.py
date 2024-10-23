import pygame
import random
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import pickle
import os
import sys

# Constants

GRID_SIZE = 4
WIDTH, HEIGHT = GRID_SIZE*50, GRID_SIZE*50
TILE_SIZE = WIDTH // GRID_SIZE
FONT_SIZE = 30
BACKGROUND_COLOR = (187, 173, 160)
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (255, 100, 30),
    8192: (255, 50, 20),
    16384: (255, 20, 0),
    32768: (255, 20, 0),
    65536: (255, 20, 0),
    131072: (255, 20, 0),
    262144: (255, 20, 0),
    524288: (255, 20, 0),
    1048576: (255, 20, 0)
}

import numpy as np
import random
import pygame

class Game2048:
    def __init__(self, player_type='human'):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.spawn_tile()
        self.spawn_tile()
        self.player_type = player_type
        self.moves = {pygame.K_UP: 'up', pygame.K_DOWN: 'down', pygame.K_LEFT: 'left', pygame.K_RIGHT: 'right'}
        self.score = 0  # Initialize score

    def spawn_tile(self):
        empty_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if self.grid[r, c] == 0]
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.grid[row, col] = 2 if random.random() < 0.9 else 4

    def move(self, direction):
        old_grid = self.grid.copy()

        if direction == 'up':
            self.grid = self.move_up()
        elif direction == 'down':
            self.grid = self.move_down()
        elif direction == 'left':
            self.grid = self.move_left()
        elif direction == 'right':
            self.grid = self.move_right()

        if not np.array_equal(old_grid, self.grid):
            self.spawn_tile()

    def merge_row(self, row):
        new_row = [i for i in row if i != 0]  # Keep only non-zero tiles
        merged_row = []
        skip = False
        
        for i in range(len(new_row)):
            if skip:
                skip = False
                continue
            if i + 1 < len(new_row) and new_row[i] == new_row[i + 1]:
                merged_row.append(new_row[i] * 2)
                self.score += new_row[i] * 2  # Increase score
                skip = True
            else:
                merged_row.append(new_row[i])

        # Fill the rest of the row with zeros
        merged_row.extend([0] * (GRID_SIZE - len(merged_row)))
        return np.array(merged_row)

    def move_left(self):
        new_grid = np.array([self.merge_row(row) for row in self.grid])
        return new_grid

    def move_right(self):
        new_grid = np.array([self.merge_row(row[::-1])[::-1] for row in self.grid])
        return new_grid

    def move_up(self):
        new_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for col in range(GRID_SIZE):
            column = self.grid[:, col]
            new_grid[:, col] = self.merge_row(column)
        return new_grid

    def move_down(self):
        new_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for col in range(GRID_SIZE):
            column = self.grid[:, col][::-1]  # Reverse the column for down movement
            new_grid[:, col] = self.merge_row(column)[::-1]  # Merge and reverse back
        return new_grid

    def is_game_over(self):
        return not any(self.is_valid_move(move) for move in ['up', 'down', 'left', 'right'])

    def is_valid_move(self, direction):
        temp_grid = self.grid.copy()
        if direction == 'up':
            return not np.array_equal(temp_grid, self.move_up())
        elif direction == 'down':
            return not np.array_equal(temp_grid, self.move_down())
        elif direction == 'left':
            return not np.array_equal(temp_grid, self.move_left())
        elif direction == 'right':
            return not np.array_equal(temp_grid, self.move_right())

    def best_move(self, move_history):
        moves = ['up', 'down', 'left', 'right']
        number = random.randint(0,3)
        return moves[number] # random now
        def calculate_reward(old_grid, new_grid, action, move_history):
            reward = 0
            combinereward = 0
            # Reward for combining tiles
            combined_tiles = np.sum(old_grid) - np.sum(new_grid)
            if combined_tiles > 0:
                combinereward = combined_tiles * 10  # Reward for combining tiles

            # Penalty for overcrowding
            if np.count_nonzero(new_grid) > GRID_SIZE * GRID_SIZE * 0.75:  
                reward += 2 * (np.count_nonzero(new_grid) - GRID_SIZE * GRID_SIZE)  # Penalize for having too many tiles

            # Count how many tiles have moved
            tiles_moved = np.sum(old_grid != new_grid)
            if tiles_moved > 0:
                reward += tiles_moved * 1.2  # Reward for moving multiple tiles

            # Check for repeated moves
            if len(move_history) >= 4:
                # Count the occurrences of each action
                move_counts = {action: move_history.count(action) for action in set(move_history)}
                
                # Calculate the percentage of the most common move
                max_move_count = max(move_counts.values())
                percentage_repeated = max_move_count / len(move_history)


                
                # Reward for less frequent moves
                total_moves = len(move_history)
                if total_moves > 12:
                    frequency_count = move_counts.get(action, 0)
                    frequency_percentage = frequency_count / total_moves

                    if frequency_percentage < 0.0632:  # If this move is less than 25% of total moves
                        reward += combinereward * 2.5  # Reward for making less frequent moves

            # Update move history
            move_history.append(action)
            if len(move_history) > 4:  # Keep only the last four moves for checking
                move_history.pop(0)

            return reward

        best_action = None
        best_reward = float('-inf')  # Initialize best reward to negative infinity

        for action in ['up', 'down', 'left', 'right']:
            reward = 0
            # Simulate the move
            if self.is_valid_move(action) and action == 'up':
                new_grid = self.move_up()
            elif self.is_valid_move(action) and action == 'down':
                new_grid = self.move_down()
            elif self.is_valid_move(action) and action == 'left':
                new_grid = self.move_left()
            elif self.is_valid_move(action) and action == 'right':
                new_grid = self.move_right()
            mhc = move_history
            # Calculate the reward for the new grid
            if self.is_valid_move(action) :
                reward = calculate_reward(self.grid, new_grid, action, mhc)  # Use your reward function


            # Check if this action is the best so far
            if reward >= best_reward:
                best_reward = reward
                best_action = action

        return best_action  # Return the action that yields the highest reward

    def get_board_state(self):
        return self.grid.copy()

    def get_score(self):
        return self.score  # Return the current score


class BotPlayer:
    def __init__(self, filename=None):
        self.q_table = {}
        self.filename = filename
        self.learning_rate = 0.1  # Alpha
        self.discount_factor = 0.985  # Gamma
        self.currentDiscount = 1
        self.exploration_rate = 1.0  # Epsilon
        self.exploration_decay = 0.999  # Decay factor for exploration
        self.min_exploration_rate = 0.1

        self.previous_sum = 0  # Initialize previous sum for efficiency calculation
        self.moves_count = 0  # Initialize move count
        
        if filename and os.path.exists(filename):
            self.load(filename)
        
    def get_state_key(self, grid):
        return str(grid.reshape(GRID_SIZE * GRID_SIZE))

    def choose_move(self, game, move_history):
        state_key = self.get_state_key(game.get_board_state())
        
        # Exploration vs. Exploitation
        if random.random() < self.exploration_rate:
            return game.best_move(move_history)
        else:
            # Exploitation: Choose the best action based on Q-values
            q_values = self.q_table.get(state_key, {})
            valid_moves = {move: q_values.get(move, 0) for move in ['up', 'down', 'left', 'right'] if game.is_valid_move(move)}
            if valid_moves:
                return max(valid_moves, key=valid_moves.get)  # Choose the move with highest Q-value
            else:
                return None  # No valid moves

    def learn(self, old_state, action, reward, new_state, n_steps=3):
        """ Multi-step Q-learning to allow the bot to look ahead. """
        old_state_key = self.get_state_key(old_state)
        new_state_key = self.get_state_key(new_state)

        # Initialize Q-values if they don't exist
        if old_state_key not in self.q_table:
            self.q_table[old_state_key] = {}
        if new_state_key not in self.q_table:
            self.q_table[new_state_key] = {}

        old_q_value = self.q_table[old_state_key].get(action, 0)
        max_future_q = max(self.q_table[new_state_key].values(), default=0)

        # Update Q-value considering n future steps to allow strategizing
        future_reward = self.discount_factor ** n_steps * max_future_q
        new_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + future_reward)
        self.q_table[old_state_key][action] = new_q_value

        # Decay exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay


    def calculate_reward(self, old_grid, new_grid, action, move_history):
        reward = 0
        combinereward = 0
        # Reward for combining tiles
        combined_tiles = np.sum(old_grid) - np.sum(new_grid)
        if combined_tiles > 0:
            combinereward = combined_tiles * 10  # Reward for combining tiles

        # Penalty for overcrowding
        if np.count_nonzero(new_grid) > GRID_SIZE * GRID_SIZE * 0.75:  
            reward += 2 * (np.count_nonzero(new_grid) - GRID_SIZE * GRID_SIZE)  # Penalize for having too many tiles

        # Count how many tiles have moved
        tiles_moved = np.sum(old_grid != new_grid)
        if tiles_moved > 0:
            reward += tiles_moved * 1.2  # Reward for moving multiple tiles

        # Check for repeated moves
        if len(move_history) >= 4:
            # Count the occurrences of each action
            move_counts = {action: move_history.count(action) for action in set(move_history)}
            
            # Calculate the percentage of the most common move
            max_move_count = max(move_counts.values())
            percentage_repeated = max_move_count / len(move_history)


            
            # Reward for less frequent moves
            total_moves = len(move_history)
            if total_moves > 12:
                frequency_count = move_counts.get(action, 0)
                frequency_percentage = frequency_count / total_moves

                if frequency_percentage < 0.0632:  # If this move is less than 25% of total moves
                    reward += combinereward * 2.5  # Reward for making less frequent moves

        # Update move history
        move_history.append(action)
        if len(move_history) > 4:  # Keep only the last four moves for checking
            move_history.pop(0)

        reward = reward * self.currentDiscount
        self.currentDiscount = self.currentDiscount * self.discount_factor
        return reward




    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)


def draw_grid(screen, grid):
    screen.fill(BACKGROUND_COLOR)
    font = pygame.font.Font(None, FONT_SIZE)
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            tile_value = grid[r, c]
            rect = pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, TILE_COLORS[tile_value], rect)
            
            if tile_value != 0:
                text = font.render(str(tile_value), True, (119, 110, 101))
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
                
    pygame.display.update()

def start_game(player_type, game_number, bot_player, bot_file=None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2048 Game")
    clock = pygame.time.Clock()

    game = Game2048(player_type)

    max_tile_achieved = 0  # Initialize to keep track of max tile
    running = True
    moves = 0
    move_history = []  # Initialize move history

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if game.player_type == 'human' and event.key in game.moves:
                    game.move(game.moves[event.key])
                    move_history.append(game.moves[event.key])  # Record move

        if game.player_type == 'bot' and bot_player:
            old_grid = game.get_board_state()
            move = bot_player.choose_move(game, move_history)
            if move:
                game.move(move)
                reward = bot_player.calculate_reward(old_grid, game.get_board_state(), move, move_history)  # Pass move history
                bot_player.learn(old_grid, move, reward, game.get_board_state())
                moves += 1


        draw_grid(screen, game.grid)


        if game.is_game_over():
            game_number = game_number + 1
            max_tile_achieved = np.max(game.grid)  # Get the max tile
            print("Game ", game_number, "max tile: ", max_tile_achieved, " | Eff: ", (np.log2(max_tile_achieved) / np.log2(moves)))
            running = False






    clock.tick(6000) # 6000 = 10 ganes/sec

    return max_tile_achieved  # Return max tile achieved

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, string):
        self.widget.configure(state="normal")
        self.widget.insert("end", string, (self.tag,))
        self.widget.configure(state="disabled")

import matplotlib.pyplot as plt

def main_menu():
    def on_start():
        player_type = player_var.get()
        bot_file = bot_file_var.get() if player_type == 'bot' else None
        iterations = int(iteration_entry.get())
        max_tiles = []  # List to store max tiles from each iteration
        bot_player = BotPlayer(bot_file) if player_type == 'bot' else None

        for i in range(iterations):
            max_tile = start_game(player_type, i, bot_player, bot_file)  # Assuming start_game returns the max tile
            max_tiles.append(np.log2(max_tile))


        if(bot_player):
            bot_player.save(bot_file)
        print(f"Max Tiles from {iterations} iterations: {max_tiles}")  # Print or display max tiles
        plot_max_tiles(max_tiles)

    def plot_max_tiles(max_tiles):
        powers_of_two = [max(tile, 1) for tile in max_tiles]  # Avoid log(0) issues
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(max_tiles)), powers_of_two, marker='o', linestyle='-', color='b')
        plt.title('Max Powers of Two Achieved Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Max Power of Two')
        plt.xticks(range(len(max_tiles)))  # Set x-ticks to the iteration numbers
        plt.grid()
        plt.show()

    def load_bot():
        filename = filedialog.askopenfilename(title="Select Bot File", filetypes=[("Pickle Files", "*.pkl")])
        if filename:
            bot_file_var.set(filename)

    def create_bot():
        filename = simpledialog.askstring("Bot Filename", "Enter a filename to save the bot data:")
        if filename:
            filename += ".pkl"
            if not os.path.exists(filename):
                bot_player = BotPlayer()
                bot_player.save(filename)
                messagebox.showinfo("Success", f"Bot data saved as {filename}.")
            else:
                messagebox.showwarning("Warning", "File already exists!")


    root = tk.Tk()
    root.title("2048 Menu")

    player_var = tk.StringVar(value='human')
    bot_file_var = tk.StringVar()

    tk.Label(root, text="Choose Player Type:").pack()
    tk.Radiobutton(root, text="Human", variable=player_var, value='human').pack()
    tk.Radiobutton(root, text="Bot", variable=player_var, value='bot').pack()

    tk.Label(root, text="Number of Iterations:").pack()
    iteration_entry = tk.Entry(root)
    iteration_entry.pack()

    tk.Button(root, text="Load Bot Data", command=load_bot).pack()
    tk.Button(root, text="Create New Bot Data", command=create_bot).pack()
    tk.Button(root, text="Start Game", command=on_start).pack()
    text = tk.Text(root, wrap="word").pack()
    sys.stdout = TextRedirector(text, "stdout")
    sys.stderr = TextRedirector(text, "stderr")
    root.mainloop()
if __name__ == "__main__":
    main_menu()
