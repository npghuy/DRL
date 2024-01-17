import pfrl
from pfrl.policies import SoftmaxCategoricalHead
import torch
import torch.nn as nn
import gym
import numpy as np
import time
from pfrl.wrappers import atari_wrappers

# env = gym.make('CartPole-v1')
env = gym.make("ALE/Pong-v5")

env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, 
                                      screen_size=84, terminal_on_life_loss=False, 
                                      grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)

env = gym.wrappers.FrameStack(env, 4)

env = atari_wrappers.ClipRewardEnv(env)
state = env.reset()[0]

# obs_size = env.observation_space.low.size
obs_size = env.observation_space.low.shape[0]
n_actions = env.action_space.n

model = nn.Sequential(
    nn.Conv2d(obs_size, 16, 8, stride=4),
    nn.ReLU(),
    nn.Conv2d(16, 32, 4, stride=2),
    nn.ReLU(),
    nn.Flatten(start_dim=1),
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

def phi(x):
    return np.asarray(x, dtype=np.float32) / 255

agent = pfrl.agents.A2C(
   model,
   optimizer=torch.optim.Adam(model.parameters(), eps=1e-2),
   gamma=0.9,
   num_processes = 4,
   gpu=0,
   phi=phi
)


n_episodes = 20000
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

agent.load('save/checkpoint/a2c')
eval_time=0
# with open("./save/result/a2c.txt",'w',1,encoding="utf-8") as f:
    # start = time.time()

for i in range(1, n_episodes + 1):
    obs = env.reset()[0]
    R = 0  # return (sum of rewards)
    t = 0  # time step
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
        agent.save('save/checkpoint/a2c')
            
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
