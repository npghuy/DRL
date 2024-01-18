import numpy as np
import random

class Map:
    best_visited=[]
    best_reward=0

    def __init__(self,first_city=None,matrix=None):
        self.cur_city=first_city
        self.matrix=matrix
        self.visited=[first_city]
        self.nor=1

    def reset(self):
        self.cur_city=random.randint(0,self.matrix.shape[0]-1)
        self.visited=[self.cur_city]
        state=self.matrix[self.cur_city]
        return state
    def check_done(self):
        return len(self.visited)!=len(set(self.visited)) or len(self.visited)==self.matrix.shape[0]

    def step(self,next_city:int):
        self.visited.append(next_city)
        reward= -1 if self.visited.index(next_city)<len(self.visited)-1 else 2-self.matrix[self.cur_city][next_city]/self.nor
        self.cur_city=next_city
        state=self.matrix[self.cur_city]
        done=self.check_done()
        return state,reward,done

    def read_file(self,file_name):
        matrix=[]
        with open(file_name,'r',encoding='utf-8') as f:
            for line in f:
                row=[float(i) for i in line[:-1].split()]
                matrix.append(row)
        matrix=np.array(matrix)
        self.nor=np.max(matrix)+1
        # matrix=matrix/self.nor
        
        self.matrix=matrix
        # for i in range(self.matrix.shape[0]):
        #         self.matrix[i][i]=100000

    def sample(self):
        val_next=[]
        for i in range(self.matrix.shape[0]):
          if i not in self.visited:
              val_next.append(i)
        if val_next:
            return val_next[random.randint(0,len(val_next)-1)]
        
        return self.visited[0]
    
    def cur_score(self,visited=None):
        if visited==None:
            visited=self.visited
        cur_score=0
        for i in range(len(visited)-1):
            cur_score+=self.matrix[visited[i]][visited[i+1]]
        return cur_score
        # return cur_score*self.nor
    def best_score(self):
        cur_score=0
        if not self.best_visited:
            return 0
        for i in range(len(self.best_visited)-1):
            cur_score+=self.matrix[self.best_visited[i]][self.best_visited[i+1]]
        cur_score+=self.matrix[self.best_visited[-1]][self.best_visited[0]]
        return cur_score
        # return cur_score*self.nor

Saleman=Map()
Saleman.read_file('../Dataset/p01_d.txt')  
# print(Saleman.matrix) 

import pfrl
import torch
import torch.nn as nn
import gym
import numpy as np
import time
from pfrl.wrappers import atari_wrappers

Saleman.reset()

obs_size=n_actions=Saleman.matrix.shape[0]

class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)
        return pfrl.action_value.DiscreteActionValue(h)

q_func = QFunction(obs_size, n_actions)

optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

gamma = 0.9

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=Saleman.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

def phi(x):
    # Feature extractor
    return np.asarray(x, dtype=np.float32)

gpu = 0
agent=pfrl.agents.DQN(q_func,optimizer,replay_buffer,gamma,explorer,gpu=gpu,replay_start_size=500,update_interval=1,target_update_interval=100,phi=phi)

n_episodes = 5000000
max_episode_len = 500000
# agent.load('../Checkpoint/dqn')
with open("../Results/dqn.txt",'w',1) as f:
    start = time.time()
    for i in range(1, n_episodes + 1):
        obs = Saleman.reset()
        R = 0  # return (sum of rewards)
        t = 0  # time step
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            # action = agent.act(np.reshape(obs,-1))
            try:
                action = agent.act(obs)
            except:
                print(obs)
                action = agent.act(obs)
            obs, reward, done= Saleman.step(action)
            R += reward
            t += 1
            reset = t == max_episode_len
            # agent.observe(np.reshape(obs,-1), reward, done, reset)
            # print(obs,reward,done,action,sep='\n')
            agent.observe(obs, reward, done, reset)

            if done or reset:
                if len(Saleman.visited)>=Saleman.matrix.shape[0]:
                    if Saleman.best_reward<R:
                        Saleman.best_reward=R
                        Saleman.best_visited=Saleman.visited
                break
        if i % 1000 == 0:
            print('episode:', i, 'R:', R)
            print("num stopped points: ",len(Saleman.visited))
            print("best_score: ",Saleman.best_score())
            if Saleman.best_reward!=0:
                if Saleman.best_score()<=291:
                    print('success!',Saleman.best_score())
                    break
            print('statistics:', agent.get_statistics())
            agent.save('../Checkpoint/dqn')
            f.write(str(i)+"\t"+str(Saleman.best_score())+"\t"+str(Saleman.best_visited)+'\n')
    end = time.time()
    f.write(str(end-start)+"\n")
    print("Running time: ",end - start)

print('Finished.')