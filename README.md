# FinCatch AI Challenge

Implementation of the hard version of the FinCatch programming assessment.

## Contents
mlp_for_llm_sim.py - q1
agent.py - q2 agent
market_sim.py - q3 enviroment
train.py - reinforced learning 

# Discussion on the question
When the policy network is not pre-trained, convergence during reinforcement learning can be very challenging. This is mainly due to the following reasons:

Sparse Rewards. The reward is only given when the agent achieves a perfect three-digit match. With such sparse feedback, the network rarely receives a strong learning signal. Without pre-training, the agent is essentially learning from almost zero reward for many episodes, which makes it hard to adjust its policy effectively.

Large and Discrete Action Space. The action space comprises 1000 possible actions (from 1 to 1000). Learning to pick the correct action in such a high-dimensional discrete space from scratch is difficult, especially when the correct actions are only rewarded in rare circumstances.

Non-Differentiable Reward Structure. The reward mechanism relies on digit matching—a process that is not differentiable. This means that even if the agent’s actions are close to optimal, there’s no smooth gradient to guide the updates in its parameters.

To address these challenges, several techniques can be applied:

Imitation Learning or Pre-training. Use a set of “ideal” examples (e.g., known correct actions for given LLM outputs) to pre-train the network. This can give the agent a good starting point before applying reinforcement learning.

Reward Shaping. Modify the reward function to provide intermediate rewards. For example, instead of only rewarding a perfect three-digit match, you might provide small rewards for achieving one or two matching digits. This can help guide the network’s learning more effectively.

Advanced RL Algorithms. Consider using actor-critic methods or other RL algorithms like Proximal Policy Optimization (PPO) that can handle sparse rewards better than a simple REINFORCE approach.


# Thoughts on task
Should the LLM output be normalized to a consistent four-digit format for all inputs—even those between 10 and 100—to ensure that the reward mechanism (which relies on matching three digits) can function properly, given that single-digit outputs make matching impossible?

"""
Test Case 2:
Input 1 → LLM → state = 1111 → Agent → Action = 199 → Environment: returns the
number of matching digits → State = 2 → LLM → state = 2222 → Agent → Action = 213
→ Environment: returns the number of matching digits → State = 1 → LLM → state = 1111
→ Agent → Action = 111 → Environment: returns the number of matching digits → State
= 3 → LLM → state = 3333 → Agent → Action = 333 → Environment: returns the number
of matching digits → State = 3 →→Done: state matches the previous result → If state = 3,
return reward = 100/4, where 4 represents the number of interactions with the environment.
"""

In Test Case 2, when the LLM outputs 1111 and the agent selects 199, only the first digit should match (expected count = 1), but the system reports 2. Notably, in other cases, the state matches the previous matches and would have ended the loop. This inconsistency raises the question: should the matching state be used as the input for the next step, or is there another mechanism intended to maintain consistency?

When there are zero matches, should the next market input be set to zero? An input of zero may not trigger the intended special-case behavior (like outputting 1111–9999), which could lead to further inconsistencies in the LLM output.

If the current state is zero and it matches the previous state, should the episode be terminated? Terminating on a zero state might be ambiguous since it never yields a reward, disrupting the intended feedback mechanism.