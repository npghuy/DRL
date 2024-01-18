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

def make_env(test:bool):
    # process_seed=int(process_seeds[idx])
    # env_seed=2**32-1 -process_seed if test else train_seed
    env = gym.make("ALE/Pong-v5")
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, 
                                      screen_size=84, terminal_on_life_loss=False, 
                                      grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    
    env = gym.wrappers.FrameStack(env, 4)
    if not test:
        env = atari_wrappers.ClipRewardEnv(env)
    # env.seed(env_seed)
    return env

env=make_env(False)
obs_size = env.observation_space.low.shape[0]

n_actions = env.action_space.n
# n_actions=env.action_space.low.size

# Normalize observations based on their empirical mean and variance
# obs_normalizer = pfrl.nn.EmpiricalNormalization(
#     obs_size, clip_threshold=5
# )

policy = torch.nn.Sequential(
    nn.Conv2d(obs_size, 16, 8, stride=4),
    nn.Tanh(),
    nn.Conv2d(16, 32, 4, stride=2),
    nn.Tanh(),
    nn.Flatten(),
    nn.Linear(2592, 256),
    nn.Tanh(),
    nn.Linear(256, n_actions),
    # SoftmaxCategoricalHead(),
    pfrl.policies.GaussianHeadWithStateIndependentCovariance(
        action_size=n_actions,
        var_type="diagonal",
        var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
        var_param_init=0,  # log std = 0 => std = 1
    ),
)

vf = torch.nn.Sequential(
    nn.Conv2d(obs_size, 16, 8, stride=4),
    nn.Tanh(),
    nn.Conv2d(16, 32, 4, stride=2),
    nn.Tanh(),
    nn.Flatten(),
    nn.Linear(2592, 256),
    nn.Tanh(),
    nn.Linear(256, 1),
)

# While the original paper initialized weights by normal distribution,
# we use orthogonal initialization as the latest openai/baselines does.
def ortho_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)

ortho_init(policy[0], gain=1)
ortho_init(policy[2], gain=1)
# ortho_init(policy[4], gain=1e-2)
ortho_init(policy[5], gain=1)
ortho_init(policy[7], gain=1e-2)
ortho_init(vf[0], gain=1)
ortho_init(vf[2], gain=1)
# ortho_init(vf[4], gain=1e-2)
ortho_init(vf[5], gain=1)
ortho_init(vf[7], gain=1e-2)

def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32) / 255


# TRPO's policy is optimized via CG and line search, so it doesn't require
# an Optimizer. Only the value function needs it.
vf_opt = torch.optim.Adam(vf.parameters(), lr=2.5e-4, eps=1e-5)

# Hyperparameters in http://arxiv.org/abs/1709.06560
agent = pfrl.agents.TRPO(
    policy=policy,
    vf=vf,
    vf_optimizer=vf_opt,
    # obs_normalizer=obs_normalizer,
    gpu=0,
    update_interval=5000,
    max_kl=0.01,
    conjugate_gradient_max_iter=20,
    conjugate_gradient_damping=1e-1,
    gamma=0.995,
    lambd=0.97,
    vf_epochs=5,
    entropy_coef=0,phi=phi
)

n_episodes = 8000
max_episode_len = 500000
def eval(n=3):
    ave_R=0
    with agent.eval_mode():
        for i in range(n):
            obs = env.reset()[0]
            R = 0  
            t = 0
            while True:
                action = np.argmax(agent.act(obs))
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

# agent.load('../Checkpoint/trpo')
eval_time=0

with open("../Results/trpo.txt",'w',1) as f:
    start = time.time()

    for i in range(1, n_episodes + 1):
        obs = env.reset()[0]
        R = 0  # return (sum of rewards)
        t = 0  # time step
        while True:
            # print(obs)
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = np.argmax(agent.act(obs))
            
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
            agent.save('../Checkpoint/trpo')

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






















