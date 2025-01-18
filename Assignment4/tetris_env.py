import numpy as np
from random import choices

#All possible shapes of Tetris
shapes = {
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    'S': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}
shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']

# checks if shape is on already occupied position
def is_occupied(shape, anchor, board):
    abs_shape = [(i + anchor[0], j + anchor[1]) for i,j in shape]
    for x, y in abs_shape:
        if y < 0:
            continue
        if x < 0 or x > board.shape[1] - 1 or y > board.shape[0] - 1 or board[y, x]:
            return True
    return False


# functions for actions
def idle(shape, anchor, board):
    return shape, anchor
def move_left(shape, anchor, board):
    new_anchor = anchor[0] -1, anchor[1]
    return (shape, new_anchor) if not is_occupied(shape, new_anchor, board) else (shape, anchor)
def move_right(shape, anchor, board):
    new_anchor = anchor[0] + 1, anchor[1]
    return (shape, new_anchor) if not is_occupied(shape, new_anchor, board) else (shape, anchor)
def move_down(shape, anchor, board):
    new_anchor = anchor[0], anchor[1] + 1
    return (shape, new_anchor) if not is_occupied(shape, new_anchor, board) else (shape, anchor)
def rotate_left(shape, anchor, board):
    new_shape = [(j, -i) for i,j in shape]
    return (new_shape, anchor) if not is_occupied(new_shape, anchor, board) else (shape, anchor)
def rotate_right(shape, anchor, board):
    new_shape = [(-j, i) for i, j in shape]
    return (new_shape, anchor) if not is_occupied(new_shape, anchor, board) else (shape, anchor)

#Tetris Environment
class TetrisEnv:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=bool) # static board which has collided shapes, i.e, no longer moving
        self.view_board = self.board.copy() # dynamic board having shape which is currently moving
        self.score = 0
        self.actions = {
            1: 'move_left',
            2: 'move_right',
            3: 'move_down',
            0: 'idle',
            4: 'rotate_left',
            5: 'rotate_right',
        }
        self.action_map = {
            1: move_left,
            2: move_right,
            3: move_down,
            0: idle,
            4: rotate_left,
            5: rotate_right,
        }
        self.n_actions = len(self.action_map)
        self.prob_arr = [10]*len(shapes) # dynamic probability for generating shapes

    #generates random shape
    def generate_shape(self):
        r = choices([0,1,2,3,4,5,6], self.prob_arr)[0]
        self.prob_arr[r] -= 1
        if self.prob_arr[r] == 0:
            self.prob_arr[r] = 1
        return shapes[shape_names[r]]

    #adds shapes to environment
    def add_shape_to_env(self):
        self.anchor = (self.width//2 - 1, 0)
        self.shape = self.generate_shape()

    # checks if shape has collided
    def has_collided(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    # sets shape on game board
    def set_shape_on_board(self,board,f=True):
        for i, j in self.shape:
            x, y = self.anchor[0] + i, self.anchor[1] + j
            if 0 <= x < self.width and 0 <= y < self.height:
                board[y, x] = f

    # clears board lines if possible and updates score
    def clear_lines(self):
        cleared = 0
        for y in range(self.height - 1, -1, -1):
            if np.all(self.board[y, :]):  # Check if the row is completely filled
                cleared += 1
                self.board[1:y + 1, :] = self.board[:y, :]  # Shift everything above down
                self.board[0, :] = False  # Clear the top row
        if cleared > 0:
            print('cleared')
            self.score += 10*cleared
        return cleared

    # to reduce gaps which are impossible to fill
    def blocked_gaps(self):
        blocked_gaps = 0
        for i in range(self.width):
            arr_unfilled = np.where(self.board[:, i] == False)[0]
            arr_gaps = []

            # Find gaps by checking for non-consecutive indices
            for k in range(len(arr_unfilled) - 1):
                if arr_unfilled[k] != arr_unfilled[k + 1] - 1:
                    arr_gaps.append(arr_unfilled[k + 1] - 1)

            # Check if gaps are blocked
            for j in arr_gaps:
                if i == 0 or i == self.width - 1:
                    blocked_gaps += 1  # Gaps on the edges are blocked
                elif (
                        i > 0 and i < self.width - 1 and
                        self.board[j, i - 1] and self.board[j, i + 1]
                ):
                    blocked_gaps += 1  # Gaps surrounded by filled spaces are blocked

        #print('Blocked gaps:', blocked_gaps)
        return blocked_gaps

    #Tried rewards in env
    '''
    def consecutive_filled(self):
        r = 0
        for i in range(self.height):
            lengths = []
            count = 0
            for num in self.board[i, :]:
                if num == 1:
                    count += 1
                else:
                    if count > 0:
                        lengths.append(count)
                    count = 0
            if count > 0:
                lengths.append(count)
            if len(lengths) != 0:
                r += max(lengths) / self.width * (i / self.height) ** 4
        return r

    def bumpiness_filled(self):
        arr = []
        for i in range(self.width):
            arrs = np.where(self.board[:, i] == True)[0]
            if len(arrs) == 0:
                arr.append(0)
            else:
                arr.append(self.height - min(arrs))
        arr = np.array(arr)
        sd = np.std(arr)
        return sd'''

    #reward to reduce unnecessary stacking of blocks
    def ratio_row_filled(self):
        r = 0
        for i in range(self.height):
            sum = np.sum(self.board[i, :])
            r += (sum/self.width) * (i / self.height) ** 4
        return r

    # to guide for better placing of shapes
    def heatmap_array(self):
        rows, cols = self.view_board.shape
        x = np.arange(cols)
        y = np.arange(rows)
        x_grid, y_grid = np.meshgrid(x, y)

        distances = np.sqrt(((x_grid - 0) ** 2) / 3 + (y_grid - 0) ** 2)
        decay_factor = 0.2
        gradient = np.exp(-decay_factor * distances)
        gradient = gradient + gradient[:,::-1] # makes a gradient having different weights for different position
        normalized_gradient = (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient)) # making it normalized
        r = -2*(np.linalg.norm(normalized_gradient-self.board))**0.5
        return r

    def step(self, action):
        reward = 0
        #print(self.actions[action])
        score =self.score
        old_shape = self.shape
        old_anchor = self.anchor
        self.shape, self.anchor = self.action_map[action](self.shape, self.anchor, self.board)
        if old_shape == self.shape and old_anchor == self.anchor: # reduces useless actions
            reward -= 1
        self.shape, self.anchor = move_down(self.shape, self.anchor, self.board)
        done = False
        if self.has_collided():
            #reward += 1
            #print(self.bumpiness_filled())
            #reward -= 3 * self.bumpiness_filled()
            reward += self.ratio_row_filled()
            #print('collided')
            self.set_shape_on_board(self.board,True)
            reward -= self.heatmap_array()
            reward += 100* self.clear_lines()
            if np.any(self.board[0, :]):
                print('Game Over')
                reward -= 100
                reward -= 5 * self.blocked_gaps()
                score = self.score
                self.clear()
                done = True
            else:
                self.add_shape_to_env()

        self.view_board = self.board.copy()
        self.set_shape_on_board(self.view_board,True)
        return self.view_board, reward, done, score

    def clear(self):
        self.score = 0
        self.add_shape_to_env()
        self.board = np.zeros_like(self.board)
        return self.view_board