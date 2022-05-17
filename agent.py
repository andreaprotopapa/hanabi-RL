import numpy as np

class Agent():
    def __init__(self, initial_state, actions, epsilon=0.2, gamma=0.8, alpha=0.1, load_learned = False, save_filename="learned_qTable.npy"):
        """ Q-Learning Agent, used to choose the best action based on the best reward. 

        Args:
            initial_state (tuple): initial state of the enviroment
            actions (list): possible actions on the actual state.
            epsilon (float, optional): the percentage of how much you want to explore. Defaults to 0.2.
            gamma (float, optional): discount factor. Defaults to 0.8.
            alpha (float, optional): LR. Defaults to 0.1.
            load_learned (bool, optional): use the Q-Table already learned in past trainings. Defaults to False.
            save_filename (str, optional): file where save/load the Q-Table. Defaults to "learned_qTable.py".
        """

        self.q_table = {initial_state: np.zeros(len(actions),dtype=float)} # Initialize q-table as a dictionary (sparse table)
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.actions= actions

        if load_learned == True:
            try:
                self.load_learned_model(save_filename)
            except:
                print("No saved model found.")
                print("Q-Table reset")
        else:
            print("Q-Table reset")

    def pick_action(self, state):
        # state: actual state
        action = None

        random_probability = np.random.rand()

        if(random_probability<self.epsilon) or (not np.any(list(self.q_table[state]))): # if Exploration: select a random action
            action = self.actions[np.random.randint(low=0,high=len(self.actions))] # Select random action

        else: # Act greedy: Exploit: select the action with max value (future reward)
            action = self.actions[np.argmax(self.q_table[state])] # Select best action based on value
        return action

    def update_q_table(self,state,action,new_state,reward,is_terminal):
        action = self.actions.index(action)
        if is_terminal==True:
            self.q_table[state][action] = (1-self.alpha)*self.q_table[state][action] + self.alpha*(reward - self.q_table[state][action]) 
        else:
            self.q_table[state][action] = (1-self.alpha)*self.q_table[state][action] + self.alpha*(reward + self.gamma*np.max(self.q_table[new_state]) - self.q_table[state][action]) 

    def load_learned_model(self, filename):
        loaded_q_table = np.load(filename, allow_pickle='TRUE')
        self.q_table = loaded_q_table.item()
        print("Learned values from Q-Table loaded")

    def save_learned_model(self, filename):
        np.save(filename,self.q_table)
        print("Learned values from Q-Table saved")