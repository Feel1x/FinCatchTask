# FinCatch AI Challenge

Implementation of the hard version of the FinCatch programming assessment.

## Contents
- **mlp_for_llm_sim.py** – Q1: MLP for simulating market analysis.
- **agent.py** – Q2: AgentPolicyNetwork for trading decisions.
- **market_sim.py** – Q3: Environment simulating market interactions.
- **train.py** – Reinforcement learning training loop.
- **finance.py** – Integration with real market data using yfinance.

## Overview
This repository contains a prototype trading agent built using reinforcement learning. The system consists of three main components:
• An MLP that simulates market analysis (LLM simulation).
• An agent that converts the market analysis into a 3-digit trading action.
• A market environment that simulates interactions and computes rewards based on digit matching.
The training loop applies the REINFORCE algorithm with entropy regularization and uses the Adam optimizer for adaptive learning rates.

### Discussion on the question
My training results reveal that the agent struggles to achieve meaningful rewards over 5000 episodes. Throughout all episodes, the total reward remained at 0.00, indicating that the agent consistently failed to reach the target state (state = 3) within the allowed interactions. Additionally, the weight norm of the policy network showed only minimal changes, suggesting that the updates during training were insufficient to guide the agent toward high-reward actions. This stagnation is primarily due to the sparse reward structure, where rewards are only given for achieving three matching digits, combined with the large action space, which makes exploration inefficient and rare successes even harder to achieve.

The challenges are compounded by the nature of the task itself. The reward mechanism relies on digit matching, a process that lacks differentiability, meaning there’s no smooth gradient to guide parameter updates. Without intermediate rewards or pre-training, the agent essentially learns from almost zero feedback for many episodes, making it nearly impossible to adjust its policy effectively. Furthermore, the action space consists of 1,000 possible actions, which significantly increases the difficulty of finding the correct action in such a high-dimensional discrete space.

To address these issues, several techniques can be applied to improve performance. For instance, imitation learning or pre-training the agent using synthetic data with known correct actions for given LLM outputs could provide a solid starting point before applying reinforcement learning. Reward shaping is another promising approach, where intermediate rewards are introduced for partial successes, such as achieving one or two matching digits, to create a smoother learning signal. Advanced reinforcement learning algorithms like Proximal Policy Optimization (PPO) or actor-critic methods could also be employed, as they are better suited for handling sparse rewards and improving exploration efficiency compared to a simple REINFORCE approach.

By implementing these strategies, the agent’s performance can be significantly enhanced. Pre-training provides a foundation, reward shaping encourages incremental progress, and advanced RL algorithms ensure more effective exploration and learning. Together, these methods address the core challenges of the task and offer a path toward more efficient training and decision-making.


### Thoughts on task
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

When there are zero matches, should the next market input be set to zero? An input of zero may not trigger the intended special-case behavior (like outputting 1111–9999), which could lead to further inconsistencies in the LLM output. In my implementation, when there are zero matches, the next market input defaults to 1 or random to avoid ambiguity.

If the current state is zero and it matches the previous state, should the episode be terminated? Terminating on a zero state might be ambiguous since it never yields a reward, disrupting the intended feedback mechanism.