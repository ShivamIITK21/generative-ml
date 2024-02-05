import math
import random as r
import requests
import base64

def toB64(s: str):
    return base64.b64encode(s.encode("ascii"))

class NormalGaussian:
    def __init__(self):
        pass
    
    def sample(self) -> float:
        z1 = r.uniform(0, 1)
        z2 = r.uniform(0, 1)

        y1 = math.sqrt(-2*math.log(z1))*(math.cos(2*math.pi*z2))
        return y1

class Gaussian(NormalGaussian):
    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance
        self.sd = math.sqrt(variance)

    def sample(self) -> float:
        s = super().sample()
        return self.sd*s + self.mean 

if __name__ == "__main__":
    dist = Gaussian(1, 2)
    counts = {}
    # requests.get(url = "http://localhost:8080/add", params={"id": toB64("norm"), "exp": toB64(r"""\frac{e^{\frac{-x^2}{2}}}{\sqrt{2\pi}}""")})
    for _ in range(0, 1000):
        v = dist.sample()
        rounded = round(10*v)/10
        if rounded not in counts: 
            counts[rounded] = 1
        else :
            counts[rounded] += 1
    scounts = dict(sorted(counts.items()))
    mean = sum([k*v for (k, v) in scounts.items()])/1000 
    var = sum([v*(k-mean)*(k-mean) for (k, v) in scounts.items()])
    var1 = var/1000 
    var2 = var/(1000-1)
    print(scounts)
    print(mean)
    print(var1)
    print(var2)
