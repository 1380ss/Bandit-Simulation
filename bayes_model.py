"""
this file contains all Bayes updating models.
in the future can check if there's packages available
"""


import numpy as np

class BetaBernoulli:
    """
    Beta-Bernoulli model
    """

    def __init__(self, number_of_arms, prior=None):
        """
        only initialize prior (posterior will be set later, when number of replication
        is known
        :param number_of_arms:
        :param prior:
        """
        self.number_of_arms = number_of_arms
        if prior is None:
            self.prior = {'a': np.ones(number_of_arms),
                          'b': np.ones(number_of_arms)}
        else:
            self.prior = prior
        self.posterior = {'a': None,
                          'b': None}

    def get_posterior_sample(self, size=1):
        """
        get (multiple) posterior sample using current 'self.posterior'
        :param size: number of posterior samples
        :return:
        """

        return np.random.beta(**self.posterior, size=(size,) + self.posterior['a'].shape)

    def update_posterior(self, action_hist, reward_hist):
        """
        update posterior distribution based on reward and action history.
        This needs to be called at the beginning to initialize posterior
        :param action_hist:
        :param reward_hist:
        :return:
        """

        b = action_hist.shape[0]
        reward_counts = np.transpose(np.array(
            [np.sum(reward_hist * (action_hist == i), axis=1) for i in range(self.number_of_arms)]
        ))
        action_counts = np.transpose(np.array(
            [np.sum((action_hist == i), axis=1) for i in range(self.number_of_arms)]
        ))
        self.posterior['a'] = np.tile(self.prior['a'], (b, 1)) + reward_counts
        self.posterior['b'] = np.tile(self.prior['b'], (b, 1)) + action_counts - reward_counts
