import torch
import numpy as np
import pygame
from environment import PongDoublesEnv
from model import PPOAgent

def play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PongDoublesEnv(render_mode="human")
    agent = PPOAgent().to(device)
    
    try:
        checkpoint = torch.load("pong_doubles_weights.pt", map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            agent.load_state_dict(checkpoint['model_state'])
        else:
            agent.load_state_dict(checkpoint)
        print("Model loaded.")
    except:
        print("No weights found.")
        return

    agent.eval()
    obs_dict, _ = env.reset()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return

        obs_tensor = torch.tensor(np.array([obs_dict[a] for a in env.agents]), dtype=torch.float32).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor, deterministic=True)
        
        obs_dict, _, terms, _, _ = env.step({a: action[i].item() for i, a in enumerate(env.agents)})
        env.render()
        if any(terms.values()): obs_dict, _ = env.reset()

if __name__ == "__main__":
    play()