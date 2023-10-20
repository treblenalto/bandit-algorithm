import numpy as np
import pandas as pd


class Greedy:
    """Greedy algorithm

    Select the arm with the highest value(mean reward).
    """

    def __init__(self, arms: int = 2):
        self.arms = arms
        self.count = [0] * arms
        self.values = [0] * arms

    def select(self):
        return self.values.index(max(self.values))

    def update(self, arm: int, reward: float):
        self.count[arm] += 1
        self.values[arm] = (
            (self.count[arm] - 1) * self.values[arm] + reward
        ) / self.count[arm]


class EpsilonGreedy:
    """Epsilon-Greedy algorithm

    Select the arm with the highest value(mean reward) with probability 1-epsilon.
    """

    def __init__(self, epsilon: float = 0.1, arms: int = 2):
        self.epsilon = epsilon
        self.arms = arms
        self.count = [0] * arms
        self.values = [0] * arms

    def select(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.arms)
        else:
            return self.values.index(max(self.values))

    def update(self, arm: int, reward: float):
        self.count[arm] += 1
        self.values[arm] = (
            (self.count[arm] - 1) * self.values[arm] + reward
        ) / self.count[arm]


class UCB:
    """Upper Confidence Bound(UCB) algorithm

    Select the arm maximizing the upper confidence bound.
    """

    def __init__(self, c: int = 1, arms: int = 2):
        self.c = c
        self.arms = arms
        self.count = [0] * arms
        self.values = [0] * arms

    def ucb(self, arm: int):
        if sum(self.count) < self.arms:
            return np.inf
        else:
            return self.values[arm] + self.c * np.sqrt(
                np.log1p(sum(self.count)) / self.count[arm]
            )

    def select(self):
        if sum(self.count) < self.arms:
            return sum(self.count)
        else:
            u = [self.ucb(arm) for arm in range(self.arms)]
            return u.index(max(u))
    
    def update(self, arm: int, reward: float):
        self.count[arm] += 1
        self.values[arm] = (
            (self.count[arm] - 1) * self.values[arm] + reward
        ) / self.count[arm]
        
class ThompsonSampling:
    """Thompson Sampling algorithm

    Select the arm based on the probability of the arm being the best.
    """

    def __init__(self, arms: int = 2):
        self.arms = arms
        self.alpha = np.ones(arms)
        self.beta = np.ones(arms)
        self.count = [0] * arms
        self.values = [0] * arms
    
    def select(self):
        samples = np.random.beta(self.alpha, self.beta)
        self.count[np.argmax(samples)] += 1
        return np.argmax(samples)
    
    def update(self, arm: int, reward: float):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
        self.values[arm] = (
            (self.count[arm] - 1) * self.values[arm] + reward
        ) / self.count[arm]

if __name__ == "__main__":
    results_list = []
    arms = 2  # A, B

    for arm_a in np.arange(0.1, 1.0, 0.1):
        for arm_b in np.arange(0.1, 1.0, 0.1):
            for trial in range(5):
                total_reward = 0

                for name, agent in zip(
                    ["Greedy", "EpsilonGreedy", "UCB", "ThompsonSampling"],
                    [Greedy(arms=arms), EpsilonGreedy(arms=arms), UCB(arms=arms), ThompsonSampling(arms=arms)],
                ):
                    for t in range(100):
                        arm = agent.select()
                        prob = arm_a if arm == 0 else arm_b
                        reward = np.random.binomial(1, prob)
                        total_reward += reward
                        agent.update(arm, reward)
                    total_reward = total_reward / 100
                    results_list.append([arm_a, arm_b, name, total_reward])

    results = pd.DataFrame(
        results_list, columns=["E(P(A))", "E(P(B))", "algorithm", "score"]
    )
    results.to_csv("results.csv", index=False)
    print(results.groupby("algorithm").mean()["score"])