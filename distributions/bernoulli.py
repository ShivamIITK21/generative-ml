import random as r
import bisect

class Bernoulli:
    def __init__(self, probs: list[float]) -> None:
        assert(len(probs) != 0)
        ps = sum(probs)
        assert(abs(ps-1) <= 0.01)
        self.probs = probs
        self.cumulative_probs = self.__calcCumulativeProbs() 
        
    def sample(self) -> int :
        uni_random = r.uniform(0,1)
        inx = bisect.bisect_left(self.cumulative_probs, uni_random, lo = 0, hi = len(self.cumulative_probs))
        return inx


    def __calcCumulativeProbs(self) -> list[float]:
        cprobs: list[float] = []
        sum: float = 0
        for x in self.probs:
            sum += x
            cprobs.append(sum)
        return cprobs


if __name__ == "__main__":
    probs = [0.1, 0.2, 0.3, 0.4]
    dist = Bernoulli(probs)
    counts = [0, 0, 0, 0]
    for _ in range(0, 100000):
        s = dist.sample()
        counts[s] += 1
    print(counts)
