from models.em_maximization.em_scaler import EMScaler
from distributions.gaussian_mixture import GaussianMixture
from visual_utils.visualizer import Visualizer
import time

if __name__ == "__main__":
    target = GaussianMixture([0, 1, 2], [0.1, 0.1, 0.1], [0.333, 0.333, 0.334])
    train_data = target.sampleDummy(10000)

    vis = Visualizer()
    vis.plotPoints(train_data, 10)
    data_list = ([[k]*v for (k, v) in train_data.items()])
    data = [x for xs in data_list for x in xs]
    # data = [1.0, 2.0, 3.0, 4.0]
    model = EMScaler(data, 5)
    vis.draw("model", model.gaussian_mix.getLatex())
    print(model.gaussian_mix.getAllParams())
    for i in range(0, 1000):
        print(f"{i} iteration...")
        q = model.EStep()
        new_probs, new_means, new_vars = model.MStep(q)
        model.gaussian_mix.updateBernoilliProbs(new_probs)
        model.gaussian_mix.updateGaussians(new_means, new_vars)
        vis.remove("model")
        vis.draw("model", model.gaussian_mix.getLatex())
        time.sleep(0.2)
    print(model.gaussian_mix.getAllParams())
