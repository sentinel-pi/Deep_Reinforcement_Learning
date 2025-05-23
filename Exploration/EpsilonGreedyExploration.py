import torch

def epsilon_greedy(action_values ,epsilon,exploit=False):
    #exploit phase only exploit
    if (torch.rand(1).item()) <= epsilon and not exploit:
        action = torch.randint(0,len(action_values),(1,1)).item()
    else:
        action = torch.argmax(action_values).item()
    return action 
 
def epsilon_decay(epsilon,decay_rate,min_epsilon):
    return max(epsilon*decay_rate,min_epsilon)

def linear_epsilon_decay(epsilon,decay_rate,min_epsilon,Episode=1000):
    
    return max(epsilon-decay_rate,min_epsilon)