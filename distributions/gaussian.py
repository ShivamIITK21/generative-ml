import math
import random as r
import time
from visual_utils.visualizer import Visualizer


class NormalGaussian:
    def __init__(self):
        pass
    
    def sample(self) -> float:
        z1 = r.uniform(0, 1)
        z2 = r.uniform(0, 1)

        y1 = math.sqrt(-2*math.log(z1))*(math.cos(2*math.pi*z2))
        return y1

    def getLatex(self) -> str:
        return r"""\frac{e^{\frac{-x^2}{2}}}{\sqrt{2\pi}}"""

class Gaussian(NormalGaussian):
    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance
        self.sd = math.sqrt(variance)

    def sample(self) -> float:
        s = super().sample()
        return self.sd*s + self.mean 

    def probDensity(self, x: float) -> float:
        return (1/math.sqrt(2*math.pi*self.variance))*(math.exp((-1*(x-self.mean)*(x-self.mean))/(2*self.variance)))

    def getLatex(self) -> str:
        return r"""\frac{e^{\frac{-(x - """ + f"""{self.mean}""" + r""")^2}{2*""" + f"""{self.variance}""" + r"""}}}{\sqrt{2\pi *""" + f"""{self.variance}""" + r"""}}""" 

    def getMean(self):
        return self.mean

    def getVar(self):
        return self.variance

    def visualize(self, N: int, ts: float, multiplier: int):
        visual = Visualizer()
        counts = {}
        visual.draw("norm", self.getLatex())
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


