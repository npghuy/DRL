import numpy as np
import torch
from torch import nn
import gym
import functools
from collections import deque
import time
import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.wrappers import atari_wrappers


class MultiBinaryAsDiscreteAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self.orig_action_space = env.action_space
        self.action_space = gym.spaces.Discrete(2 ** env.action_space.n)

    def action(self, action):
        return [(action >> i) % 2 for i in range(self.orig_action_space.n)]


utils.set_random_seed(0)

train_seed = 0
test_seed = 2 ** 31 - 1

def make_env(test):
    env = gym.make("ALE/Pong-v5")
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, 
                                      screen_size=84, terminal_on_life_loss=False, 
                                      grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    
    env = gym.wrappers.FrameStack(env, 4)
    env_seed = test_seed if test else train_seed
    env.seed(int(env_seed))

    if isinstance(env.action_space, gym.spaces.MultiBinary):
        env = MultiBinaryAsDiscreteAction(env)
    return env

env = make_env(test=False)
n_actions = env.action_space.n
obs_size = env.observation_space.low.shape[0]

model = nn.Sequential(
    nn.Conv2d(obs_size, 16, 8, stride=4),
    nn.ReLU(),
    nn.Conv2d(16, 32, 4, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(2592, 256),
    nn.ReLU(),
    pfrl.nn.Branched(
        nn.Sequential(
            nn.Linear(256, n_actions),
            SoftmaxCategoricalHead(),
        ),
        nn.Linear(256, 1),
    ),
)

opt = torch.optim.Adam(model.parameters(), lr=2.5e-4, eps=1e-5)

def phi(x):
    return np.asarray(x, dtype=np.float32) / 255

agent = PPO(
        model,
        opt,
        gamma=0.99,
        gpu=0,
        phi=phi,
        update_interval=128 * 8,
        minibatch_size=64,
        epochs=10,
        clip_eps=0.2,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=0,
        max_grad_norm=0.5,
    )

n_episodes = 4000
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
agent.load('save/checkpoint/ppo')
eval_time=0
# with open("./save/result/ppo.txt",'w',1) as f:
    # start = time.time()

for i in range(1, n_episodes + 1):
    obs = env.reset()[0]
    R = 0  
    t = 0
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act(obs)
        obs, reward, done_1,done_2, _ = env.step(action)
        done=(done_1 or done_2)
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
        print('statistics:', agent.get_statistics())
        agent.save('save/checkpoint/ppo')

    #         start_eval=time.time()
    #         ave_R=eval()
    #         end_eval=time.time()
    #         eval_time+=end_eval-start_eval
    #         f.write(str(i)+"\t"+str(ave_R)+'\n')
    #         if time.time()-start-eval_time>60000:
    #             break
    # end = time.time()
    # f.write(str(end-start-eval_time)+"\n")
    # print("Running time: ",end - start-eval_time)

print('Finished.')


























