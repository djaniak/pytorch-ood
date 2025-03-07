"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-yes-brightred?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.Entropy
    :members:

"""
from typing import Optional, TypeVar

from torch import Tensor, nn

from ..api import Detector

Self = TypeVar("Self")


class Entropy(Detector):
    """
    Implements Entropy-based OOD detection.

    This methods calculates the entropy based on the logits of a classifier.
    Higher entropy means more uniformly distributed posteriors, indicating larger uncertainty.
    Entropy is calculated as

    .. math::
        H(x) = - \\sum_i^C  \\sigma_i(f(x)) \\log( \\sigma_i(f(x)) )

    where :math:`\\sigma_i` indicates the :math:`i^{th}` softmax value and :math:`C` is the number of classes.

    """

    def fit(self: Self, *args, **kwargs) -> Self:
        """
        Not required.
        """
        return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        """
        Not required.
        """
        return self

    def __init__(self, model: nn.Module):
        """
        :param model: the model :math:`f`
        """
        super(Entropy, self).__init__()
        self.model = model

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate entropy for inputs

        :param x: input tensor, will be passed through model

        :return: Entropy score
        """
        return self.score(self.model(x))

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        """
        :param logits: logits of input
        """
        p = logits.softmax(dim=1)
        return -(p.log() * p).sum(dim=1)


class LatentEntropy(Detector):
    """
    Implements latent entropy-based OOD detection.

    This methods calculates the entropy based on the entropy of a latent space distribution.
    """

    def fit(self: Self, *args, **kwargs) -> Self:
        """
        Not required.
        """
        return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        """
        Not required.
        """
        return self

    def __init__(self, model: nn.Module):
        """
        :param model: the model :math:`f`
        """
        super(LatentEntropy, self).__init__()
        self.model = model

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate entropy for inputs

        :param x: input tensor, will be passed through model

        :return: Entropy score
        """
        return self.model(x).entropy()

class LatentNegativeEntropy(Detector):
    """
    Implements latent entropy-based OOD detection.

    This methods calculates the entropy based on the entropy of a latent space distribution.
    """

    def fit(self: Self, *args, **kwargs) -> Self:
        """
        Not required.
        """
        return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        """
        Not required.
        """
        return self

    def __init__(self, model: nn.Module):
        """
        :param model: the model :math:`f`
        """
        super(LatentNegativeEntropy, self).__init__()
        self.model = model

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate entropy for inputs

        :param x: input tensor, will be passed through model

        :return: Entropy score
        """
        return -self.model(x).entropy()
