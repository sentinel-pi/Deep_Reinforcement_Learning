from glm import sizeof
import numpy as np 
import torch

class ExperienceReplay():
    def __init__(self,max_value):
        self.experience_replay = {"old_state":[],"old_action":[],"reward":[],"new_state":[]}
        self.max_value = max_value
        
    def append(self,old_state,old_action,reward,new_state):
        if(self.size() >= self.max_value):
            for key in self.experience_replay.keys():
                self.experience_replay[key].pop(0)
        self.experience_replay["old_state"].append(old_state)
        self.experience_replay["old_action"].append(old_action)
        self.experience_replay["reward"].append(reward)
        self.experience_replay["new_state"].append(new_state)
        
    def sample(self,batch_size):
        self.batch ={}
        indices = np.random.choice(range(self.size()),batch_size,replace=False)
        for key in self.experience_replay.keys():
            self.batch[key]=np.array(self.experience_replay[key])[indices]
        return self.batch
    def size(self):
        return len(self.experience_replay["old_state"])