from distributions.gaussian_mixture import GaussianMixture  
import random as r
import time

from visual_utils.visualizer import Visualizer

class EMScaler:
    def __init__(self, num_latent: int):
        self.data = []
        self.num_latent = num_latent
        self.bernoulli_probs = [(1/num_latent)]*num_latent
        self.means = [r.uniform(-2, 2) for _ in range(num_latent)]
        self.vars = [r.uniform(0.01,2) for _ in range(num_latent)]
        self.gaussian_mix = GaussianMixture(self.means, self.vars, self.bernoulli_probs)

    def setData(self, data: list[float]):
        self.data = data
        
    def EStep(self):
        assert(len(self.data) != 0)
        q_z = [[0.0]*self.num_latent for _ in range(0, len(self.data))]

        # O(n*k) currently, probably could make it faster using some vectorization
        probs, means, vars = self.gaussian_mix.getAllParams() 
        for n in range(0, len(self.data)):
            p_xn = self.gaussian_mix.probDensity(self.data[n])
            for k in range(0, self.num_latent):
                q_z[n][k] = (probs[k]*self.gaussian_mix.ithGaussian(k).probDensity(self.data[n]))/p_xn

        return q_z

    def MStep(self, q_z: list[list[float]]):
        assert(len(self.data) != 0)
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

    def visualize(self, vis: Visualizer, target: GaussianMixture):
        train_data = target.sampleDummy(10000)
        vis.plotPoints(train_data, 10)
        data_list = ([[k]*v for (k, v) in train_data.items()])
        data = [x for xs in data_list for x in xs]
        self.setData(data)
        vis.draw("model",self.gaussian_mix.getLatex())
        for i in range(0, 1000):
            print(f"{i} iteration...")
            q = self.EStep()
            new_probs, new_means, new_vars = self.MStep(q)
            self.gaussian_mix.updateBernoilliProbs(new_probs)
            self.gaussian_mix.updateGaussians(new_means, new_vars)
            vis.remove("model")
            vis.draw("model", self.gaussian_mix.getLatex())
            time.sleep(0.2)
