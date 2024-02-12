from distributions.gaussian import Gaussian

if __name__ == "__main__":
    g = Gaussian(0, 1)
    g.visualize(1000, 0.1, 10)
