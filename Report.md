[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: imgs/reacher.gif "Reacher"
[image4]: imgs/crawler.gif "Crawler"
[image5]: imgs/scores_reacher.png "ReacherScores"
[image6]: imgs/scores_crawler.png "CrawlerScores"

## The environments


### Reacher Environment

![Trained Agent][image1]

In this environment[environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher), a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location (represented by the green sphere). Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. 

The task is considered solved once the agent gets an average score  of +30 over 100 consecutive episodes.

### Distributed Training

For this project, two separate versions of the Unity environment are provided:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience (This version is included in the repo). You can download the single agent version from this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)  

The barrier for solving the multi-agent version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).


### Reacher

In this environment, the goal is to teach a creature with four legs to walk forward without falling. This is a more difficult environment with the following characteristics:

*   Vector Observation space: 117 variables corresponding to position, rotation, velocity, and angular     velocities of each limb plus the acceleration and angular acceleration of the body.

*   Vector Action space: (Continuous) Size of 20, corresponding to target rotations for joints.


## Training

### Reacher
For the reacher, the Distributed Distributional Deep Deterministic Policy Gradient algorithm [D4PG](https://openreview.net/pdf?id=SyZipzbCb) is used, this algorithm is at its core an off-policy actor-critic method. This means that we consider a parametrized actor-function (Implemented as a DNN in the `model.py` file) which specifies the current policy by deterministically mapping states to a specific action. At the same  time, a critic that estimates the action-value function (Q(s,a)) is learned and is used to estimate the actor's loss function as a baseline for the expected Reward. 

Same as in [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), this algorithm also uses:


1. Memory Replay: Several steps are stored on a memory buffer and then randomly sampled for training as defined in the `ReplayBuffer` class on the file  `model/dqn_agent.py`, hence coping with state-action correlation.

2. Fixed Q-Targets: This is done by means of a second network (same architecture as the DQN) that is used as a target network. However instead of fixing and updating the target network every n-steps, a soft update is performed.

3. Aditionally, Batch Normalization is used

For more details, check the [D4PG paper](https://openreview.net/pdf?id=SyZipzbCb).

Training process:
![Reacher][image3]

Scores:
![Scores][image5]

#### Hyperparameters

### Crawler

For this environment, the [PPO](https://arxiv.org/pdf/1707.06347.pdf) algorithm was implemented with an actor-critic style as shown in the `model.py` and `ppo_agent.py` files.

This algorithm approximates Expected reward using the surogate function, so that previous steps, collected with an initial policy can be used for several training steps to train the policy, this surogate function is also clipped to help convergence which can be affected by this approximation. 

The agent's value function is approximated through a critic, that shares the weights of the initial layers with the actor. These values are then used to calculate advantages for the algorithm. 

The training process can be seen here:

![Crawler][image4]

Training Scores for crawler:

![ScoresCrawler][image6]


### Hyperparameters
## Reacher
The main parameters defined for the D4PG algorithm used for the Reacher environment are:
```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```
## Crawler
The main parameter used for the actor-critic style PPO algorithm were:
```python
iterations = 2000               # max number of training iterations
gamma = 0.99                    # Discound factor
nsteps = 2048                   # Number of steps collected at each training step
ratio_clip = 0.2                # Ratio used for clipping the approximate gradient (from importance sampling)
nbatchs = 32                    # Batch size
epochs = 10                     # Number of episodes collected at each training step
gradient_clip = 0.5             # Parameter that defines how to clip gradients
lrate = 2e-4                    # Initial Learning rate
lrate_schedule = lambda it: max(0.995 ** it, 0.01)    # Function defining how learning rate decays
```
## Future work
As future work, more Policy based algorithms such as [A3C](https://arxiv.org/pdf/1602.01783.pdf) or A2C could be tested for these environments. Aditionally, given the obtained results, it can be seen that the agents have to be trained on a big number of episodes, this could be optimized by performing a parameter search on parameters such as learning rate, batch size and weight decay.