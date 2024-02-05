from distributions.bernoulli import Bernoulli

if __name__ == "__main__":
    b = Bernoulli([0.5, 0.1, 0.4])
    print(b.sample())
