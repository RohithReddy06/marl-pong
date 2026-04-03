import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from environment import PongDoublesEnv
from model import PPOAgent

def train():
    # 1. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- MARL PONG: JITTER-REDUCTION MODE ---")
    print(f"Hardware: {str(device).upper()}")
    
    env = PongDoublesEnv(render_mode=None) 
    agent = PPOAgent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    checkpoint_path = "pong_doubles_weights.pt"
    
    # Auto-Resume Logic
    if os.path.exists(checkpoint_path):
        print(f"Loading existing weights...")
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Hyperparameters
    steps_per_rollout = 256
    total_iterations = 3000
    
    # Entropy Decay Config (Forces 'Certainty' over time)
    START_ENTROPY = 0.05
    END_ENTROPY = 0.01
    VF_COEF = 0.5    # Value Loss Weight

    print(f"Training started. Weights save to {checkpoint_path} on exit.")

    try:
        for iteration in range(total_iterations):
            # Calculate dynamic entropy coefficient (Linear Decay)
            # As iterations go up, randomness goes down.
            ent_coef = max(END_ENTROPY, START_ENTROPY - (iteration / 2000) * (START_ENTROPY - END_ENTROPY))
            
            obs_list, act_list, logp_list, rew_list, val_list, term_list = [], [], [], [], [], []
            curr_obs_dict, _ = env.reset()
            
            # --- DATA COLLECTION (Rollout) ---
            for _ in range(steps_per_rollout):
                obs_tensor = torch.tensor(np.array([curr_obs_dict[a] for a in env.agents]), 
                                          dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    action, logp, _, val = agent.get_action_and_value(obs_tensor)
                
                action_dict = {a: action[i].item() for i, a in enumerate(env.agents)}
                next_obs_dict, reward_dict, terms, _, _ = env.step(action_dict)
                
                obs_list.append(obs_tensor)
                act_list.append(action)
                logp_list.append(logp)
                val_list.append(val.flatten())
                rew_list.append(torch.tensor([reward_dict[a] for a in env.agents], dtype=torch.float32).to(device))
                term_list.append(torch.tensor([float(terms[a]) for a in env.agents], dtype=torch.float32).to(device))
                
                curr_obs_dict = next_obs_dict if not any(terms.values()) else env.reset()[0]

            # --- PREPARE BATCHES ---
            b_obs = torch.stack(obs_list).view(-1, 10) 
            b_actions = torch.stack(act_list).view(-1)
            b_logprobs = torch.stack(logp_list).view(-1)
            
            b_rewards_unflat = torch.stack(rew_list)
            b_values_unflat = torch.stack(val_list)
            b_terms_unflat = torch.stack(term_list)

            # Compute Advantages per Agent Correctly
            returns_unflat = torch.zeros_like(b_rewards_unflat).to(device)
            
            # BOOTSTRAP using next state's value
            next_obs_tensor = torch.tensor(np.array([curr_obs_dict[a] for a in env.agents]), dtype=torch.float32).to(device)
            with torch.no_grad():
                _, _, _, next_val = agent.get_action_and_value(next_obs_tensor)
            future_ret = next_val.view(-1)
            
            for t in reversed(range(len(b_rewards_unflat))):
                future_ret = b_rewards_unflat[t] + 0.99 * future_ret * (1.0 - b_terms_unflat[t])
                returns_unflat[t] = future_ret
            
            returns = returns_unflat.view(-1)
            b_values = b_values_unflat.view(-1)
            b_rewards = b_rewards_unflat.view(-1)
            
            advantages = returns - b_values
            # NORMALIZE Advantages (Critical for PPO stability)
            advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- PPO UPDATE ---
            for _ in range(4): 
                _, new_logp, entropy, new_val = agent.get_action_and_value(b_obs, b_actions)
                ratio = (new_logp - b_logprobs).exp()
                surr1 = ratio * advantages_norm
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_norm
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((new_val.view(-1) - returns) ** 2).mean()
                
                # Use the decaying ent_coef here
                loss = policy_loss - (ent_coef * entropy.mean()) + (VF_COEF * value_loss)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

            # --- LOGGING ---
            if iteration % 10 == 0:
                rew_per_agent = b_rewards.view(-1, 4).mean(dim=0) 
                print(f"Iter {iteration:04d} | Loss: {loss.item():.4f} | Ent_Coef: {ent_coef:.4f}")
                print(f"  Avg Rew -> P0(T): {rew_per_agent[0]:.4f} | P1(B): {rew_per_agent[1]:.4f} | "
                      f"P2(T): {rew_per_agent[2]:.4f} | P3(B): {rew_per_agent[3]:.4f}")
                print("-" * 65)

    except KeyboardInterrupt:
        print("\n[SIGINT] Manual Pause. Saving weights...")

    torch.save(agent.state_dict(), checkpoint_path)
    print("Done. Ready to Play.")

if __name__ == "__main__":
    train()