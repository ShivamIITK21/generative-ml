from distributions.gaussian import Gaussian
from visual_utils.visualizer import Visualizer
from monte_carlo_methods.rejection_sampler import RejectionSampler
import math

def dist(x : float):
    if(x**2 > 1):
        return 0
    return math.sqrt(1-x**2)

if __name__ == "__main__":
    vis = Visualizer()
    rs = RejectionSampler(dist, Gaussian(0, 0.9*0.9), 2.7)
    for _ in range(0, 10):
        print(rs.sample())
    rs.visualize(vis, 10)
