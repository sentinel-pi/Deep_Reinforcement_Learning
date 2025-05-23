import torch
from collections import deque
import numpy as np 
import random

class ExperienceReplay():
    def __init__(self,max_value):
        self.experience_replay = deque(maxlen=max_value)
        self.max_value = max_value
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
        old_state = torch.tensor([x["old_state"]for x in batch])
        old_action = torch.tensor([x["old_action"]for x in batch])
        reward = torch.tensor([x["reward"]for x in batch])
        new_state = torch.tensor([x["new_state"]for x in batch])
        done = torch.tensor([x["done"]for x in batch])
        return (old_state,old_action,reward,new_state,done)
    def size(self):
        return len(self.experience_replay)
    def __len__(self):
        return self.size()