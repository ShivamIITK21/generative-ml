from distributions.gaussian import Gaussian
from distributions.bernoulli import Bernoulli
import time
from visual_utils.visualizer import Visualizer

class GaussianMixture:
    def __init__(self, means: list[float], variances: list[float], probs: list[float]):
        self.bernouli_dist = Bernoulli(probs)
        self.gaussians_dists = [Gaussian(mean, var) for (mean, var) in zip(means, variances)]

    def sample(self) -> float:
        return self.gaussians_dists[self.bernouli_dist.sample()].sample() 
    
    def probDensity(self, x: float):
        return sum([prob*g.probDensity(x) for (prob, g) in zip(self.bernouli_dist.getProbs(), self.gaussians_dists)])

    def getLatex(self) -> str:
        parts = [f"""{prob}""" + "*" + gauss.getLatex() for (prob, gauss) in zip(self.bernouli_dist.getProbs(), self.gaussians_dists)]
        return " + ".join(parts)
    
    def ithGaussian(self, idx: int) -> Gaussian:
        return self.gaussians_dists[idx]

    def updateBernoilliProbs(self, new_probs: list[float]):
        self.bernouli_dist = Bernoulli(new_probs)
    
    def updateGaussians(self, new_means: list[float], new_vars: list[float]):
        self.gaussians_dists = [Gaussian(mean, var) for (mean, var) in zip(new_means, new_vars)]

    def getAllParams(self):
        return (self.bernouli_dist.getProbs(), [g.getMean() for g in self.gaussians_dists], [g.getVar() for g in self.gaussians_dists])

    def visualize(self, N: int, ts: float, multiplier: int):
        visual = Visualizer()
        counts = {}
        visual.draw("mg", self.getLatex())
        for i in range(1, N+1):
            v = self.sample()
            rounded = round(10*v)/10
            if rounded not in counts: 
                counts[rounded] = 1
            else :
                counts[rounded] += 1

            for k, v in counts.items():
                prob = (v/i)*multiplier
                visual.draw(str(k), visual.getLatexKV(k, prob))
            time.sleep(ts)

