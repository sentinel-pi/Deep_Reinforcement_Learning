import torch

class ExperienceReplay():
    def __init__(self,max_value):
        self.experience_replay = {"old_state":[],"old_action":[],"reward":[],"new_state":[],"done":[]}
        self.max_value = max_value
    
    def append(self,old_state,old_action,reward,new_state,done):
        if(self.size() >= self.max_value):
            for key in self.experience_replay.keys():
                self.experience_replay[key].pop(0)
        self.experience_replay["old_state"].append(old_state)
        self.experience_replay["old_action"].append(old_action)
        self.experience_replay["reward"].append(reward)
        self.experience_replay["new_state"].append(new_state)
        self.experience_replay["done"].append(done)
        
    def sample(self,batch_size):
        batch ={}
        indices = torch.randperm(self.size())[:batch_size]
        for key in self.experience_replay.keys():
            batch[key] = torch.tensor(self.experience_replay[key])[indices]
        return batch
    def size(self):
        return len(self.experience_replay["old_state"])
    def __len__(self):
        return self.size()