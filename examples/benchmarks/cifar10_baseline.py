"""

CIFAR 10
==============================

Benchmark code for CIFAR10

+-------------+-------+---------+----------+----------+
| Detector    | AUROC | AUPR-IN | AUPR-OUT | FPR95TPR |
+=============+=======+=========+==========+==========+
| KLMatching  | 88.73 | 86.95   | 85.10    | 58.74    |
+-------------+-------+---------+----------+----------+
| MaxSoftmax  | 91.85 | 88.55   | 93.57    | 28.43    |
+-------------+-------+---------+----------+----------+
| Entropy     | 92.48 | 90.16   | 93.87    | 28.29    |
+-------------+-------+---------+----------+----------+
| MaxLogit    | 93.06 | 91.44   | 93.74    | 31.18    |
+-------------+-------+---------+----------+----------+
| EnergyBased | 93.10 | 91.51   | 93.78    | 31.05    |
+-------------+-------+---------+----------+----------+
| ODIN        | 93.20 | 92.12   | 93.94    | 31.65    |
+-------------+-------+---------+----------+----------+
| ViM         | 94.49 | 93.42   | 95.34    | 23.48    |
+-------------+-------+---------+----------+----------+
| Mahalanobis | 94.87 | 93.69   | 95.79    | 21.00    |
+-------------+-------+---------+----------+----------+


"""
import pandas as pd  # additional dependency, used here for convenience
import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
)
from pytorch_ood.detector import (
    ODIN,
    EnergyBased,
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
)
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToRGB, ToUnknown

device = "cuda:0"

torch.manual_seed(123)

# %%
# Setup preprocessing
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
trans = tvt.Compose(
    [tvt.Resize(size=(32, 32)), ToRGB(), tvt.ToTensor(), tvt.Normalize(std=std, mean=mean)]
)

# %%
# Setup datasets

dataset_in_test = CIFAR10(root="data", train=False, transform=trans, download=True)

# create all OOD datasets
ood_datasets = [Textures, TinyImageNetCrop, TinyImageNetResize, LSUNCrop, LSUNResize]
datasets = {}
for ood_dataset in ood_datasets:
    dataset_out_test = ood_dataset(
        root="data", transform=trans, target_transform=ToUnknown(), download=True
    )
    test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=256)
    datasets[ood_dataset.__name__] = test_loader

# %%
# **Stage 1**: Create DNN with pre-trained weights from the Hendrycks baseline paper
print("STAGE 1: Creating a Model")
model = WideResNet(num_classes=10, pretrained="cifar10-pt").eval().to(device)

# %%
# **Stage 2**: Create OOD detector
print("STAGE 2: Creating OOD Detectors")
detectors = {}
detectors["Entropy"] = Entropy(model)
detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
detectors["Mahalanobis"] = Mahalanobis(model.features, norm_std=std, eps=0.002)
detectors["KLMatching"] = KLMatching(model)
detectors["MaxSoftmax"] = MaxSoftmax(model)
detectors["EnergyBased"] = EnergyBased(model)
detectors["MaxLogit"] = MaxLogit(model)
detectors["ODIN"] = ODIN(model, norm_std=std, eps=0.002)

# fit detectors to training data (some require this, some do not)
print(f"> Fitting {len(detectors)} detectors")
loader_in_train = DataLoader(CIFAR10(root="data", train=True, transform=trans), batch_size=256)
for name, detector in detectors.items():
    print(f"--> Fitting {name}")
    detector.fit(loader_in_train, device=device)

# %%
# **Stage 3**: Evaluate Detectors
print(f"STAGE 3: Evaluating {len(detectors)} detectors on {len(datasets)} datasets.")
results = []

with torch.no_grad():
    for detector_name, detector in detectors.items():
        print(f"> Evaluating {detector_name}")
        for dataset_name, loader in datasets.items():
            print(f"--> {dataset_name}")
            metrics = OODMetrics()
            for x, y in loader:
                metrics.update(detector(x.to(device)), y.to(device))

            r = {"Detector": detector_name, "Dataset": dataset_name}
            r.update(metrics.compute())
            results.append(r)

# calculate mean scores over all datasets, use percent
df = pd.DataFrame(results)
mean_scores = df.groupby("Detector").mean() * 100
print(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
