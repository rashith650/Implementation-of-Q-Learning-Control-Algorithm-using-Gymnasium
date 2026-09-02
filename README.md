# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment. The agent must learn an optimal action-value function through interaction with the environment and select suitable actions to reach the goal state while avoiding the holes.

---

## Software Requirements

Programming Language: Python  
Library: Gymnasium  
Numerical Computation: NumPy  
Visualization: Matplotlib  
Environment: FrozenLake-v1  
Platform: Jupyter Notebook / Google Colab

---

## Environment Description

The FrozenLake-v1 environment is a grid-world reinforcement learning environment provided by Gymnasium.

The environment consists of a 4 × 4 grid containing:

S – Starting state  
F – Frozen/safe surface  
H – Hole, which terminates the episode  
G – Goal state

The agent starts from the initial state and must navigate through the frozen surface to reach the goal without falling into a hole.

The environment has 16 states and 4 possible actions:

| Action | Direction |
|---|---|
| 0 | Left |
| 1 | Down |
| 2 | Right |
| 3 | Up |

The agent receives a reward when it successfully reaches the goal. Falling into a hole terminates the episode.

---

## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $\max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm

Initialize the FrozenLake environment.

Initialize the Q-table with zeros for all state-action pairs.

Set the learning rate ($\alpha$), discount factor ($\gamma$), exploration rate ($\epsilon$), and other hyperparameters.

Reset the environment at the beginning of each episode.

Select an action using the epsilon-greedy policy.

Execute the selected action in the environment.

Observe the next state, reward, and termination status.

Calculate the Q-learning target:

$$
Target = R + \gamma \max_a Q(S',a)
$$

Update the Q-value using:

$$
Q(S,A) \leftarrow Q(S,A) +
\alpha[Target-Q(S,A)]
$$

Move to the next state.

Continue until the episode terminates.

Reduce epsilon gradually to decrease exploration and increase exploitation.

Repeat the process for the specified number of episodes.

Obtain the state-value function using:

$$
V(S)=\max_a Q(S,a)
$$

Obtain the learned policy by selecting the action having the maximum Q-value for each state.

Plot the learning curve and calculate the average reward over the last 1000 episodes.

---

## Python Program

```python
# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
# Write your code here

episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Q-learning target: reward + gamma * max_a Q(next_state, a)
        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] += learning_rate * (target - Q[state, action])

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Extract the greedy value function and policy after training.
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

## Python Program

```python
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Q-learning target: reward + gamma * max_a Q(next_state, a)
        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] += learning_rate * (target - Q[state, action])

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Extract the greedy value function and policy after training.
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)





```
---

## Output

## Final Q-table:


<img width="256" height="376" alt="image" src="https://github.com/user-attachments/assets/2071fb01-cbbd-4abd-885d-31e7da46f052" />





## Estimated State-Value Function:


<img width="226" height="78" alt="image" src="https://github.com/user-attachments/assets/073fa823-7c70-4ed9-b532-1f8c60432ed1" />





## Learned Policy:

<img width="307" height="87" alt="image" src="https://github.com/user-attachments/assets/a80091dd-1fd1-4315-923e-209da75c2a4a" />




## Average reward over last 1000 episodes: 
<img width="365" height="27" alt="image" src="https://github.com/user-attachments/assets/50fe6d64-8bdb-44b3-9eac-b7e0b1b3414a" />
<img width="742" height="470" alt="image" src="https://github.com/user-attachments/assets/5ac64c12-ce0f-4ec2-ac1e-43957927fe2f" />



---

## Result

```text

The Q-Learning algorithm successfully learns an action-value function for the FrozenLake environment.

The resulting Q-table contains the learned values for each state-action pair. The state-value function is obtained by taking the maximum Q-value for each state, and the learned policy selects the action with the highest Q-value.

The learning curve shows the improvement in the agent's performance during training, and the average reward over the last 1000 episodes indicates the final performance of the learned agent.

```

---

## Inference

```text

The experiment demonstrates that Q-Learning can learn an optimal action-selection policy through trial and error without requiring a predefined model of the environment.

Initially, the agent explores different actions using a high epsilon value. As training progresses, epsilon decreases, allowing the agent to increasingly exploit the learned Q-values.

The Q-table gradually learns the usefulness of different actions in each state. The learned state-value function represents the estimated value of each state, while the learned policy chooses the action with the highest estimated Q-value.

Therefore, the experiment shows that Q-Learning is effective for learning a suitable policy in the FrozenLake environment by balancing exploration and exploitation.

```

---

