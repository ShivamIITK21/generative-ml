from models.em_maximization.em_scaler import EMScaler

if __name__ == "__main__":
    e = EMScaler([0.1, 0.2, 0.3, 0.4, 0.5], 2)
    print(e.EStep())
