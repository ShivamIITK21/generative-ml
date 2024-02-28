from distributions.gaussian_mixture import GaussianMixture  
import random as r

class EMScaler:
    def __init__(self, data: list[float], num_latent: int):
        self.data = data
        self.num_latent = num_latent
        self.bernoulli_probs = [(1/num_latent)]*num_latent
        self.means = [r.uniform(-2, 2) for _ in range(num_latent)]
        self.vars = [r.uniform(0,2) for _ in range(num_latent)]
        self.gaussian_mix = GaussianMixture(self.means, self.vars, self.bernoulli_probs)

    def EStep(self):
        q_z = [[0.0]*self.num_latent for _ in range(0, len(self.data))]

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
        new_means = []
        for k in range(0, self.num_latent):
            u_k = 0
            for n in range(0, len(self.data)):
                u_k += (q_z[n][k]*self.data[n])/Nk[k]
            new_means.append(u_k)

        new_vars = []
        for k in range(0, self.num_latent):
            sigma_k = 0
            for n in range(0, len(self.data)):
                sigma_k += (q_z[n][k]*(self.data[n] - new_means[k])*(self.data[n] - new_means[k]))/Nk[k]
            new_vars.append(sigma_k)

        return (new_probs, new_means, new_vars)

    def train(self, iters: int):
        for _ in range(0, iters):
            q_z = self.EStep()
            new_probs, new_means, new_vars = self.MStep(q_z)
            self.gaussian_mix.updateBernoilliProbs(new_probs)
            self.gaussian_mix.updateGaussians(new_means, new_vars)
