from unityagents import UnityEnvironment
import numpy as np

class Environment:
    def __init__(self, train_mode = True, env_path = "Reacher_Linux_20/Reacher.x86_64"):
        self.env_base = UnityEnvironment(file_name = env_path)
        self.brain_name = self.env_base.brain_names[0]
        self.brain = self.env_base.brains[self.brain_name]
        self.train_mode = train_mode
        self.action_size = self.brain.vector_action_space_size
        self.reset()
        self.num_agents = len(self.env_info.agents)
        self.state_size = self.states.shape[1]
        print("Reacher environment Initialized")
        print(self.states.shape)
        print('Number of agents:', self.num_agents)
        print('Size of each action:', self.action_size)
        print('There are {} agents. Each observes a state with length: {}'.format(self.states.shape[0], self.state_size))
            
    def update_state(self):
        self.states = self.env_info.vector_observations

    def reset(self):
        self.env_info = self.env_base.reset(train_mode=self.train_mode)[self.brain_name]
        self.update_state()
        return self.states

    def render(self):
        pass

    def step(self, actions):
        self.env_info = self.env_base.step(actions)[self.brain_name]  # send the action to the environment
        self.update_state()
        rewards = self.env_info.rewards  # get the rewards
        dones = self.env_info.local_done  # see if episode has finished
        return self.states, rewards, dones, None #info is none

    def close(self):
        self.env_base.close()