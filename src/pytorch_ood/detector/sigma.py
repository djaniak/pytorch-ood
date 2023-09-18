from pytorch_ood.api import Detector


class HSigmaMean(Detector):
    def __init__(self, model: torch.nn.Module):
        super(HSigmaMean, self).__init__()
        self.model = model

    def fit(self, *args, **kwargs) -> "HSigmaMean":
        return self

    def fit_features(self, *args, **kwargs) -> "HSigmaMean":
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, log_var, _ = self.model(x)
        sigma = torch.exp(0.5 * log_var)
        return sigma.mean(dim=1)


class HSigmaStd(Detector):
    def __init__(self, model: torch.nn.Module):
        super(HSigmaStd, self).__init__()
        self.model = model

    def fit(self, *args, **kwargs) -> "HSigmaStd":
        return self

    def fit_features(self, *args, **kwargs) -> "HSigmaStd":
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, log_var, _ = self.model(x)
        sigma = torch.exp(0.5 * log_var)
        return -sigma.std(dim=1)


class ZSigmaMean(Detector):
    def __init__(self, model: torch.nn.Module):
        super(ZSigmaMean, self).__init__()
        self.model = model

    def fit(self, *args, **kwargs) -> "ZSigmaMean":
        return self

    def fit_features(self, *args, **kwargs) -> "ZSigmaMean":
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model(x)
        _, _, _, log_var, _ = self.model.project(h)
        sigma = torch.exp(0.5 * log_var)
        return sigma.mean(dim=1)


class ZSigmaStd(Detector):
    def __init__(self, model: torch.nn.Module):
        super(ZSigmaStd, self).__init__()
        self.model = model

    def fit(self, *args, **kwargs) -> "ZSigmaStd":
        return self

    def fit_features(self, *args, **kwargs) -> "ZSigmaStd":
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model(x)
        _, _, _, log_var, _ = self.model.project(h)
        sigma = torch.exp(0.5 * log_var)
        return -sigma.std(dim=1)
