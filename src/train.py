import torch
import torch.optim as optim
from mlp_for_llm_sim import MLP
from agent import AgentPolicyNetwork
from market_sim import MarketEnvironment


"""
Train a trading agent using REINFORCE with entropy regularization.
Uses an MLP for market analysis, an AgentPolicyNetwork to convert LLM output into a trading action,
and a MarketEnvironment to simulate market interactions and compute rewards by digit matching.
"""


def train_agent(mlp, agent, env, episodes=5000, lr=1e-2):
    # Adam is used as the optimizer; it adapts the learning rate for each parameter during training.
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    for episode in range(episodes):
        env.reset()
        market_input = 1
        episode_log_probs = []
        total_reward = 0
        steps = 0

        # Capture norm before the episode's training begins
        norm_before = agent.fc1.weight.data.norm().item()

        while True:
            steps += 1
            llm_output, action, state, reward, done, next_input, log_prob = env.step(market_input)
            episode_log_probs.append(log_prob)
            total_reward += reward
            market_input = next_input
            if done:
                break

        # Compute REINFORCE loss
        R = total_reward
        loss = -sum(episode_log_probs) * R  # Standard REINFORCE loss

        # Add entropy regularization 
        entropy = -torch.exp(log_prob) * log_prob  # Entropy formula: -p * log(p)
        loss -= 0.01 * entropy.sum()  # Add entropy term to the loss

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Capture norm after the update
        norm_after = agent.fc1.weight.data.norm().item()

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward: {total_reward:.2f}, Steps: {steps}")
            print(f"   Norm before: {norm_before:.4f}, Norm after: {norm_after:.4f}")

if __name__ == "__main__":
    mlp = MLP()
    agent = AgentPolicyNetwork()
    env = MarketEnvironment(mlp, agent)
    print("Training agent")
    train_agent(mlp, agent, env, episodes=5000, lr=1e-3)