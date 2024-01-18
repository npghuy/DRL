import numpy as np
import torch
import gym

import pfrl
from pfrl import agents, experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers, utils
from pfrl.q_functions import DistributionalDuelingDQN
from pfrl.wrappers import atari_wrappers
import time

utils.set_random_seed(0)

train_seed=0
test_seed=2**32-1

def make_env(test):
    env = gym.make("ALE/Pong-v5")
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, 
                                      screen_size=84, terminal_on_life_loss=False, 
                                      grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)

    env_seed = test_seed if test else train_seed
    env = gym.wrappers.FrameStack(env, 4)

    if not test:
        env = atari_wrappers.ClipRewardEnv(env)
    # obs, info = env.reset(seed=int(env_seed))
    # if test:
    #     env = pfrl.wrappers.RandomizeAction(env, 0)
    return env

env=make_env(False)
enval_env=make_env(True)

n_actions=env.action_space.n

n_atoms = 51
v_max = 10
v_min = -10
q_func = DistributionalDuelingDQN(
    n_actions,
    n_atoms,
    v_min,
    v_max,
    n_input_channels=4
)

# Noisy nets
pnn.to_factorized_noisy(q_func, sigma_scale=0.5)

# Turn off explorer
explorer = explorers.Greedy()

# Use the same hyper parameters as https://arxiv.org/abs/1710.02298
opt = torch.optim.Adam(q_func.parameters(), 6.25e-5, eps=1.5 * 10**-4)

# Prioritized Replay
update_interval = 4
steps=5 * 10**7
betasteps = steps / update_interval
rbuf = replay_buffers.PrioritizedReplayBuffer(
    10**6,
    alpha=0.5,
    beta0=0.4,
    betasteps=betasteps,
    num_steps=3,
    normalize_by_max="memory",
)

def phi(x):
    return np.asarray(x, dtype=np.float32) / 255

Agent = agents.CategoricalDoubleDQN
agent = Agent(
    q_func,
    opt,
    rbuf,
    gpu=0,
    gamma=0.99,
    explorer=explorer,
    minibatch_size=1,
    replay_start_size=2 * 10**4,
    target_update_interval=32000,
    update_interval=update_interval,
    batch_accumulator="mean",
    phi=phi,
)

n_episodes = 9000
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

# agent.load('../Checkpoint/rainbow')
eval_time=0
with open("../Results/rainbow.txt",'w',1,encoding="utf-8") as f:
    start = time.time()
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
            agent.save('../Checkpoint/rainbow')
            
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

