from models.em_maximization.em_scaler import EMScaler
from distributions.gaussian_mixture import GaussianMixture
from visual_utils.visualizer import Visualizer

if __name__ == "__main__":
    target = GaussianMixture([0, 1, 2], [0.1, 0.1, 0.1], [0.333, 0.333, 0.334])
    model = EMScaler(5)
    vis = Visualizer()
    model.visualize(vis, target)
