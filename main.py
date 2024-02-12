from distributions.bernoulli import Bernoulli
import numpy as np

if __name__ == "__main__":
    a = np.array([1,2,3])
    b = Bernoulli([0.5, 0.1, 0.4])
    print(b.sample())
