import random as r

class Uniform:
    def __init__(self, lower: float, higher: float):
        self.lower = lower
        self.higher = higher

    def sample(self) -> float:
        return r.uniform(self.lower, self.higher)

    def probDensity(self, _: float) -> float:
        return 1/(self.higher-self.lower)

    def getLatex(self):
        pass

