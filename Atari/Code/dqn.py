import pfrl
import torch
import torch.nn as nn
import gym
import numpy as np
import time
from pfrl.wrappers import atari_wrappers

env = gym.make("ALE/Pong-v5")

env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, 
                                      screen_size=84, terminal_on_life_loss=False, 
                                      grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)

env = gym.wrappers.FrameStack(env, 4)

env = atari_wrappers.ClipRewardEnv(env)
state = env.reset()[0]

class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = nn.Conv2d(obs_size, 16, 8, stride=4)
        self.l2 = nn.Conv2d(16, 32, 4, stride=2)
        self.flat=nn.Flatten()
        self.l3 = nn.Linear(2592, 256)
        self.l4 = nn.Linear(256, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h= self.flat(h)
        h = torch.nn.functional.relu(self.l3(h))
        h= self.l4(h)
        return pfrl.action_value.DiscreteActionValue(h)

obs_size = env.observation_space.low.shape[0]
# obs_size = env.observation_space.low.size
n_actions = env.action_space.n

q_func = QFunction(obs_size, n_actions)

optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

gamma = 0.9

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255

gpu = 0
agent=pfrl.agents.DQN(q_func,optimizer,replay_buffer,gamma,explorer,gpu=gpu,replay_start_size=500,update_interval=1,target_update_interval=100,phi=phi)

n_episodes = 3000
max_episode_len = 500000

def eval(n=3):
    ave_R=0
    with agent.eval_mode():
        for i in range(n):
            obs = env.reset()[0]
            R = 0  
            t = 0
            
            while True:
                action = agent.act(obs)
                obs, r, done,_, _ = env.step(action)
                R += r
                t += 1
                reset = t == max_episode_len
                agent.observe(obs, r, done, reset)

                if done or reset:
                    break
            ave_R+=R
        ave_R=ave_R//n
        print('evaluation-averR:', ave_R)
    return ave_R

eval_time=0
# agent.load('../Checkpoint/dqn')
with open("../Results/dqn.txt",'w',1) as f:
    start = time.time()
    for i in range(1, n_episodes + 1):
        obs = env.reset()[0]
        R = 0  # return (sum of rewards)
        t = 0  # time step
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            # action = agent.act(np.reshape(obs,-1))

            action = agent.act(obs)
            obs, reward, done_1,done_2, _ = env.step(action)
            done=(done_1 or done_2)
            R += reward
            t += 1
            reset = t == max_episode_len
            # agent.observe(np.reshape(obs,-1), reward, done, reset)
            agent.observe(obs, reward, done, reset)
            if done or reset:
                break
        if i % 10 == 0:
            print('episode:', i, 'R:', R)
            print('statistics:', agent.get_statistics())
            agent.save('../Checkpoint/dqn')



            start_eval=time.time()
            ave_R=eval()
            end_eval=time.time()
            eval_time+=end_eval-start_eval
            f.write(str(i)+"\t"+str(ave_R)+'\n')
            if time.time()-start-eval_time>60000:
                break
    end = time.time()
    f.write(str(end-start-eval_time)+"\n")
    print("Running time: ",end - start-eval_time)

print('Finished.')


