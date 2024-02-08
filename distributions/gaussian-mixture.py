from gaussian import Gaussian
from bernoulli import Bernoulli
import time
from visual_utils.visualizer import Visualizer

class GaussianMixture:
    def __init__(self, means: list[float], variances: list[float], probs: list[float]):
        self.bernouli_dist = Bernoulli(probs)
        self.gaussians_dists = [Gaussian(mean, var) for (mean, var) in zip(means, variances)]

    def sample(self) -> float:
        return self.gaussians_dists[self.bernouli_dist.sample()].sample() 

    def getLatex(self) -> str:
        parts = [f"""{prob}""" + "*" + gauss.getLatex() for (prob, gauss) in zip(self.bernouli_dist.getProbs(), self.gaussians_dists)]
        return " + ".join(parts)
    
    def visualize(self, N: int, ts: float, multiplier: int):
        visual = Visualizer()
        counts = {}
        visual.draw("mg", self.getLatex())
        for i in range(1, N+1):
            v = dist.sample()
            rounded = round(10*v)/10
            if rounded not in counts: 
                counts[rounded] = 1
            else :
                counts[rounded] += 1

            for k, v in counts.items():
                prob = (v/i)*multiplier
                visual.draw(str(k), visual.getLatexKV(k, prob))
            time.sleep(ts)

if __name__ == "__main__":
    dist = GaussianMixture([0,1,3],[0.1,0.1,0.1],[0.3,0.2,0.5])
    dist.visualize(1000, 0.1, 10)
