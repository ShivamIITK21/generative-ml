from distributions.gaussian_mixture import GaussianMixture  
import random as r

class EMScaler:
    def __init__(self, data: list[float], num_latent: int):
        self.data = data
        self.num_latent = num_latent
        self.bernoulli_probs = [(1/num_latent)]*num_latent
        self.means = [r.uniform(-2, 2)]*num_latent
        self.vars = [r.uniform(0,2)]*num_latent
        self.gaussian_mix = GaussianMixture(self.means, self.vars, self.bernoulli_probs)

    def EStep(self):
        q_z = [[0.0]*self.num_latent]*len(self.data)

        # O(n*k) currently, probably could make it faster using some vectorization
        probs, means, vars = self.gaussian_mix.getAllParams() 
        for n in range(0, len(self.data)):
            p_xn = self.gaussian_mix.probDensity(self.data[n])
            for k in range(0, self.num_latent):
                q_z[n][k] = (probs[k]*self.gaussian_mix.ithGaussian(k).probDensity(self.data[n]))/p_xn

        return q_z

    def MStep(self, q_z: list[list[float]]):
        Nk:list[float] = [sum(i) for i in zip(*q_z)]
        new_probs = [i/len(self.data) for i in Nk]
        new_means = [(1/nk)*sum([q_nk*xn for (q_nk, xn) in zip(q_n, self.data)]) for (nk, q_n) in zip(Nk, *q_z)]
        print(new_probs)
        print(new_means)

