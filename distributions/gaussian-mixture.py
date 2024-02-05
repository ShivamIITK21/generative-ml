from gaussian import Gaussian
from bernoulli import Bernoulli

class GaussianMixture():
    def __init__(self, means: list[float], variances: list[float], probs: list[float]):
        self.bernouli_dist = Bernoulli(probs)
        self.gaussians_dists = [Gaussian(mean, var) for (mean, var) in zip(means, variances)]

    def sample(self) -> float:
        return self.gaussians_dists[self.bernouli_dist.sample()].sample() 


if __name__ == "__main__":
    print("?")
    g = GaussianMixture([0, 5],[1, 1],[0.5, 0.5])
    print(g.sample())
