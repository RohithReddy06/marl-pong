import numpy as np
import pygame
from pettingzoo import ParallelEnv
from gymnasium import spaces

class PongDoublesEnv(ParallelEnv):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.width, self.height = 600, 400
        self.paddle_h = 60
        self.agents = ["p0", "p1", "p2", "p3"]
        self.observation_spaces = {a: spaces.Box(-1, 1, (10,)) for a in self.agents}
        self.action_spaces = {a: spaces.Discrete(3) for a in self.agents}
        self.last_action = {a: 0 for a in self.agents} # Track for Switch Penalty
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def _get_obs(self, agent_id):
        is_top = 1.0 if agent_id in ["p0", "p2"] else -1.0
        is_left = 1.0 if agent_id in ["p0", "p1"] else -1.0
        return np.array([
            self.ball_pos[0]/600, self.ball_pos[1]/400, 
            self.ball_vel[0]/5, self.ball_vel[1]/5,
            self.p_y["p0"]/400, self.p_y["p1"]/400, 
            self.p_y["p2"]/400, self.p_y["p3"]/400,
            is_top, is_left
        ], dtype=np.float32)

    def reset(self, seed=None):
        self.ball_pos = np.array([300.0, 200.0])
        v_y = np.random.uniform(-3.0, 3.0)
        v_x = 3.5 if np.random.rand() > 0.5 else -3.5
        self.ball_vel = np.array([v_x, v_y])
        self.p_y = {"p0": 50.0, "p1": 290.0, "p2": 50.0, "p3": 290.0}
        self.last_action = {a: 0 for a in self.agents}
        return {a: self._get_obs(a) for a in self.agents}, {}

    def step(self, actions):
        rewards = {a: 0.0 for a in self.agents}
        
        # 1. Update Paddles - Hard Mode Constraints
        for a, act in actions.items():
            if act == 1: self.p_y[a] -= 10
            if act == 2: self.p_y[a] += 10
            
            # Absolute physical barriers to prevent crossing into partner's zone
            if a in ["p0", "p2"]:  # Top Agents
                self.p_y[a] = max(0, min(140, self.p_y[a]))
            else:                  # Bottom Agents
                self.p_y[a] = max(200, min(340, self.p_y[a]))
            
            self.last_action[a] = act
        
        # 2. Physics
        self.ball_pos += self.ball_vel
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= 400: self.ball_vel[1] *= -1

        # Cap velocity to prevent tunneling
        MAX_VEL = 15.0
        if self.ball_vel[0] > MAX_VEL: self.ball_vel[0] = MAX_VEL
        elif self.ball_vel[0] < -MAX_VEL: self.ball_vel[0] = -MAX_VEL

        term = False
        midpoint = 200

        # 3. Collision Logic (Kept Zone Defense)
        if self.ball_pos[0] < 15 and self.ball_vel[0] < 0:
            for p in ["p0", "p1"]:
                if self.p_y[p] < self.ball_pos[1] < self.p_y[p] + 60:
                    self.ball_vel[0] *= -1.05
                    self.ball_pos[0] = 15
                    is_top_agent = (p == "p0")
                    is_ball_top = (self.ball_pos[1] < midpoint)
                    # HIGH REWARD for hit
                    rewards[p] += 2.0 if is_top_agent == is_ball_top else -0.5
                    break

        elif self.ball_pos[0] > 585 and self.ball_vel[0] > 0:
            for p in ["p2", "p3"]:
                if self.p_y[p] < self.ball_pos[1] < self.p_y[p] + 60:
                    self.ball_vel[0] *= -1.05
                    self.ball_pos[0] = 585
                    is_top_agent = (p == "p2")
                    is_ball_top = (self.ball_pos[1] < midpoint)
                    rewards[p] += 2.0 if is_top_agent == is_ball_top else -0.5
                    break

        # 4. Goals
        if self.ball_pos[0] <= 0:
            rewards = {a: rewards[a] + (-2.0 if a in ["p0","p1"] else 2.0) for a in self.agents}
            term = True
        elif self.ball_pos[0] >= 600:
            rewards = {a: rewards[a] + (2.0 if a in ["p0","p1"] else -2.0) for a in self.agents}
            term = True
        
        # 5. Zone Constraint + STRONGER Tracking
        if not term:
            for a in self.agents:
                is_top = a in ["p0", "p2"]
                paddle_center = self.p_y[a] + 30
                ball_y = self.ball_pos[1]
                
                # Proximity Reward (Making the ball 'magnetic')
                dist = abs(paddle_center - ball_y) / 400
                if is_top == (ball_y < midpoint):
                    rewards[a] += 0.02 * (1.0 - dist)  # Positive reward for tracking

        return {a: self._get_obs(a) for a in self.agents}, rewards, {a: term for a in self.agents}, {a: False for a in self.agents}, {}
    
    def render(self):
        if self.render_mode != "human": return
        self.screen.fill((0, 0, 0))
        pygame.draw.line(self.screen, (40, 40, 40), (0, 200), (600, 200), 1)
        pygame.draw.rect(self.screen, (255,255,255), (*self.ball_pos, 10, 10))
        for i, a in enumerate(self.agents):
            x = 10 if i < 2 else 580
            color = (100, 100, 255) if i % 2 == 0 else (255, 100, 100)
            pygame.draw.rect(self.screen, color, (x, self.p_y[a], 10, self.paddle_h))
        pygame.display.flip()
        self.clock.tick(60)