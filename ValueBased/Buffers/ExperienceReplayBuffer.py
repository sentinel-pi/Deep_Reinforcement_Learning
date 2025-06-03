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

class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self,max_value,device="cpu"):
        super().__init__(max_value,device)

        self.priorities      = deque(maxlen=max_value)
        self.alpha           = 1.0
        self.alpha_final     = 0.6
        self.alpha_decrement = 1e-3
        
        self.beta           = 0.5
        self.beta_increment = 1e-3
        self.beta_final     = 1.0
        self.epsilon        = 1e-5
        self.max_priority   = 1.0 # this is initial probability
        
    # new experiences have to have high priority too this will happen before getting Value and Target Q values
    def append(self,old_state,old_action,reward,new_state,done):
        super().append(old_state,old_action,reward,new_state,done)
        self.priorities.append(self.max_priority)
    # this happens after Append and before evaluation 
    def sample(self,batch_size):
        P = np.array(self.priorities,dtype=np.float64) + self.epsilon
        P = P ** self.alpha
        P = P / P.sum()
        N = self.__len__()
        
        IS_weights = (N*P) ** -self.beta
        IS_weights /= IS_weights.max()
        
        self.alpha = max(self.alpha_final,self.alpha - self.alpha_decrement)
        self.beta = min(self.beta_final,self.beta + self.beta_increment)
        idx = np.random.choice(range(self.__len__()),p=P , size=batch_size)
        
        IS_weights = torch.tensor(np.array([IS_weights[i] for i in idx]),device=self.device)
        batch = [self.experience_replay[i] for i in idx]
        
        old_state  = torch.tensor(np.array([x["old_state"]for x in batch]),device=self.device)
        old_action = torch.tensor(np.array([x["old_action"]for x in batch]),device=self.device)
        reward     = torch.tensor(np.array([x["reward"]for x in batch]),device=self.device)
        new_state  = torch.tensor(np.array([x["new_state"]for x in batch]),device=self.device)
        done       = torch.tensor(np.array([x["done"]for x in batch]),device=self.device)
        
        batch = (old_state,old_action,reward,new_state,done)
        
        
        return (batch,idx,IS_weights)
    # after getting q values from model and target we update our priorities with our td error 
    def update(self,index,td_priority):
        if td_priority.max().item() > self.max_priority:
            self.max_priority = td_priority.max().item() 
        for idx,p in zip(index,td_priority): 
            self.priorities[idx] =  p.item() + self.epsilon
            
class N_STEP_ExperienceReplay():
    
    def __init__(self,max_value,device="cpu",n_step = 1):
        self.experience_replay = deque(maxlen=max_value)
        self.max_value         = max_value
        self.device            = device
        self.n_step            = n_step
        random.seed(13)
    def append(self,old_state,old_action,reward,new_state,done):
        experience               = {}
        experience["old_state"]  = old_state
        experience["old_action"] = old_action
        experience["reward"]     = reward
        experience["new_state"]  = new_state
        experience["done"]       = done
        self.experience_replay.append(experience)
        
    def sample(self,batch_size):

        idx        = random.sample(range(self.size()-(self.n_step-1)),k=batch_size)
        batch      = [self.experience_replay[i] for i in idx]
        old_state  = torch.tensor(np.array([x["old_state"]for x in batch]),device=self.device)
        old_action = torch.tensor(np.array([x["old_action"]for x in batch]),device=self.device)
        
        n_step_rewards = []
        n_step_done    = []
        reward         = []
        new_state      = []
        done           = []
        for i in idx:
            n_step_rewards.clear()
            n_step_done.clear()
            done_soon = False
            for n in range(self.n_step):
                # print(f"index={i}")
                n_step_rewards.append(self.experience_replay[i+n]["reward"])
                if self.experience_replay[i+n]["done"] == True:
                    done_soon = True 
                n_step_done.append(done_soon)
            reward.append(n_step_rewards.copy())
            done.append(n_step_done.copy())
            new_state.append( self.experience_replay[i+self.n_step-1]["new_state"])
        reward    = torch.tensor(np.array(reward),device=self.device)
        new_state = torch.tensor(np.array(new_state),device=self.device)
        done      = torch.tensor(np.array(done),device=self.device)
        return (old_state,old_action,reward,new_state,done)
    def size(self):
        return len(self.experience_replay)
    def __len__(self):
        return self.size()