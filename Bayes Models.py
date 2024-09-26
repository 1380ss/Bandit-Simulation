import numpy as np

class beta_bernoulli:
    def __init__(self, number_of_arms, prior=None):
        self.number_of_arms = number_of_arms
        if prior is None:
            self.prior = {'a': np.ones(number_of_arms),
                          'b': np.ones(number_of_arms)}
        else:
            self.prior = prior
        self.posterior = {'a': None,
                          'b': None}

    def get_posterior_sample(self, size=1):

        return (np.random.beta(**self.posterior, size=(size,) + self.posterior['a'].shape))

    def update_posterior(self, action_hist, reward_hist):

        B = action_hist.shape[0]
        reward_counts = np.transpose(np.array(
            [np.sum(reward_hist * (action_hist == i), axis=1) for i in range(self.number_of_arms)]
        ))
        action_counts = np.transpose(np.array(
            [np.sum((action_hist == i), axis=1) for i in range(self.number_of_arms)]
        ))
        self.posterior['a'] = np.tile(self.prior['a'], (B, 1)) + reward_counts
        self.posterior['b'] = np.tile(self.prior['b'], (B, 1)) + action_counts - reward_counts
