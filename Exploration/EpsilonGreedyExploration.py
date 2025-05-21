import numpy as np

def epsilon_greedy(actions ,epsilon):
    if (np.random.rand()) <= epsilon:
        action = np.random.randint(0,len(actions) - 1)
    else:
        action = np.argmax(actions)
    return action 