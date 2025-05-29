import torch
from collections import deque
import numpy as np 
import random

class ExperienceReplay():
    def __init__(self,max_value,device="cpu"):
        self.experience_replay = deque(maxlen=max_value)
        self.max_value = max_value
        self.device = device 
        random.seed(13)
    def append(self,old_state,old_action,reward,new_state,done):
        experience = {}
        experience["old_state"]=old_state
        experience["old_action"]=old_action
        experience["reward"]=reward
        experience["new_state"]=new_state
        experience["done"]=done
        self.experience_replay.append(experience)
        
    def sample(self,batch_size):
        batch = random.sample(self.experience_replay,batch_size)
        old_state = torch.tensor(np.array([x["old_state"]for x in batch]),device=self.device)
        old_action = torch.tensor(np.array([x["old_action"]for x in batch]),device=self.device)
        reward = torch.tensor(np.array([x["reward"]for x in batch]),device=self.device)
        new_state = torch.tensor(np.array([x["new_state"]for x in batch]),device=self.device)
        done = torch.tensor(np.array([x["done"]for x in batch]),device=self.device)
        return (old_state,old_action,reward,new_state,done)
    def size(self):
        return len(self.experience_replay)
    def __len__(self):
        return self.size()