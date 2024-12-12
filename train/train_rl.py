import torch
import numpy as np
from torch.utils.data import DataLoader

from ppo_agent import PPOAgent

from model_rl import PointTransformerV3

class PTv3Env:


    def __init__(self, 
                 base_enc_patch_size=(1024,1024,1024,1024,1024),
                 episodes=10,
                 max_steps_per_episode=5):
        self.base_enc_patch_size = base_enc_patch_size
        self.episodes = episodes
        self.max_steps = max_steps_per_episode
        self.current_step = 0
        self.episode_count = 0
        self.done = False


        self.state = None

    def reset(self):
        self.episode_count += 1
        self.current_step = 0
        self.done = False

        self.state = np.array([np.random.uniform(0.5,2.0) for _ in range(6)], dtype=np.float32)
        return self.state

    def step(self, action):

        if action == 0:
            enc_patch_size = tuple([max(256, x-256) for x in self.base_enc_patch_size])
        elif action == 1:
            enc_patch_size = self.base_enc_patch_size
        else:
            enc_patch_size = tuple([min(2048, x+256) for x in self.base_enc_patch_size])


        model = PointTransformerV3(
            in_channels=6,
            enc_patch_size=enc_patch_size,
        ).eval()  

        N = 10000  
        data_dict = {
            "feat": torch.randn(N, 6),
            "coord": torch.randn(N, 3),
            "grid_size": 0.01,
            "offset": torch.tensor([N], dtype=torch.long)
        }


        with torch.no_grad():
            output = model(data_dict)
        

        reward = float(np.random.uniform(0,1))  


        self.current_step += 1
        if self.current_step >= self.max_steps or self.episode_count >= self.episodes:
            self.done = True

        self.state = np.array([np.random.uniform(0.5,2.0) for _ in range(6)], dtype=np.float32)

        return self.state, reward, self.done, {}

def main():

    state_dim = 6  
    action_dim = 3 
    env = PTv3Env(episodes=2, max_steps_per_episode=5)

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)


    num_episodes = 2
    for episode in range(num_episodes):
        state = env.reset()
        for step in range(env.max_steps):

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)


            agent.store_outcome(reward, done)
            state = next_state

            if done:

                agent.update()
                break

    print("训练完成")

if __name__ == "__main__":
    main()