import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

"""
AgentPolicyNetwork for making trading decisions.
Converts LLM output into a 3-digit action using a 2-layer net and 3 digit heads.
Also includes a simple reward function based on digit matching.
"""

class AgentPolicyNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        # define layers: fc1 (input_dim->hidden_dim), fc2 (hidden_dim->hidden_dim), and 3 digit heads (hidden_dim->10 each)
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.digit_heads = nn.ModuleList([nn.Linear(hidden_dim, 10) for _ in range(3)])
    
    def forward(self, x):
        # process x through fc layers with ReLU; output a list of logits from each digit head
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return [head(x) for head in self.digit_heads]
    
    def get_action(self, llm_output: int):
        # normalize llm_output, run forward, softmax & sample digits, then combine to form a 3-digit action
        x = torch.tensor([[llm_output / 10000.0]], dtype=torch.float32)
        logits = self.forward(x)
        action_digits = []
        log_probs = []
        for logit in logits:
            probs = F.softmax(logit, dim=1)
            m = torch.distributions.Categorical(probs)
            digit = m.sample()
            action_digits.append(digit.item())
            log_probs.append(m.log_prob(digit))
        action = action_digits[0] * 100 + action_digits[1] * 10 + action_digits[2]
        action = max(1, min(1000, action))
        return action, sum(log_probs)  # return final action & sum of log_probs for policy updates
    
def calc_reward(mlpOutput, agentAction):
    # compare digit counts between mlpOutput and agentAction; reward: 10 (1 match), 20 (2 matches), 100 (3 matches)
    mlp_digit = Counter(str(int(mlpOutput)))         # 313 -> Counter{'3': 2, '1': 1})
    agent_digit = Counter(str(int(agentAction)))
    
    match = 0
    for i in agent_digit:
        if i in mlp_digit:
            match += min(mlp_digit[i], agent_digit[i])

    if match == 1:
        return 10
    if match == 2:
        return 20
    if match == 3:
        return 100
    return 0


if __name__=="__main__":
    agent=AgentPolicyNetwork()
    print(f"----Test for q2----")
    print(f"Agent action: {agent.get_action(1111)[0]}")  # Random action in [1, 1000]
    print(f"Agent action: {agent.get_action(7777)[0]}")  # Random action in [1, 1000]
    print(f"Agent action: {agent.get_action(6532)[0]}")  # Random action in [1, 1000]
    print(f"Agent reward: {calc_reward(1111, 111)}")  # 3 matches → 100
    print(f"Agent reward: {calc_reward(1234, 414)}")  # 2 match → 20