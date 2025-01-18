from tetris_env import TetrisEnv as env
import torch
from model import DQN
import pygameview

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


env = env(6,12)
n_actions = env.n_actions #using env.action_space.n since action size is discrete
state = env.clear()
n_observations = env.width * env.height
model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load("policy_net_final.pth", weights_only=True)) #loading trained model
model.eval()
terminated = False
trewardinf = 0

# saving frames for pygame
frames = []
scores = []

#test run
while not terminated:
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    frames.append(env.view_board)
    action = model(state).max(1).indices.view(1,1)
    observation, reward, terminated, score = env.step(action.item())
    scores.append(score)
    trewardinf += reward
    state = observation
pygameview.pygame_view(frames, scores)
print(score)
print(trewardinf)