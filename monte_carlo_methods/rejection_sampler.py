from distributions.bernoulli import Bernoulli
from distributions.uniform import Uniform
from distributions.gaussian import Gaussian
from typing import Callable
from visual_utils.visualizer import Visualizer
import time

class RejectionSampler:
    def __init__(self, dist: Callable[[float], float], standard_dist: Gaussian|Uniform, multiplier: float):
        self.dist = dist
        self.standard_dist = standard_dist
        self.multiplier = multiplier

    def sample(self):
        while(True):
            candidate = self.standard_dist.sample()
            acceptance_prob = self.dist(candidate)/(self.multiplier*self.standard_dist.probDensity(candidate))
            if acceptance_prob == 0:
                continue
            b = Bernoulli([1-acceptance_prob, acceptance_prob])
            accept = b.sample()
            if(accept == 1):
                return candidate

    def visualize(self, vis: Visualizer, multiplier: int):
        counts = {}
        num = 0
        for _ in range(0, 10000):
            s = self.sample()
            rounded_s = round(10*s)/10
            if rounded_s not in counts:
                counts[rounded_s] = 1
            else:
                counts[rounded_s] += 1
            num += 1
            
            for (k, v) in counts.items():
                v_scaled = (v/num)*multiplier
                vis.draw(str(k), vis.getLatexKV(k, v_scaled))
             
            time.sleep(0.2)
