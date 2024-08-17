
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model

def build_optimized_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False)
    x = base_model.output
    x = UpSampling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# evaluator.py
from typing import Dict, Literal
import torch
from tabulate import tabulate
import torch.distributed as dist
import numpy as np
import tensorflow as tf

from gvcore.utils.logger import logger
import gvcore.utils.distributed as dist_utils
from gvcore.utils.structure import TensorList
from gvcore.evaluator import EVALUATOR_REGISTRY
from evaluator.api import intersect_and_union

@EVALUATOR_REGISTRY.register("segmentation")
class SegmentationEvaluator:
    """
    Evaluates segmentation models with various metrics.

    Attributes:
        num_classes (int): Number of classes in the segmentation task.
        _distributed (bool): Flag indicating if distributed evaluation is used.
        _mode (Literal["onetime", "window_avg"]): Evaluation mode.
        window_size (int): Size of the sliding window for "window_avg" mode.
    """
    def __init__(
        self, num_classes: int, distributed: bool = False, mode: Literal["onetime", "window_avg"] = "onetime", window_size: int = 0
    ):
        self.num_classes = num_classes
        self._distributed = distributed
        self._mode = mode
        self.window_size = window_size

        self._initialize_metrics()

    def _initialize_metrics(self):
        if self._mode == "onetime":
            self._total_area_intersect = torch.zeros((self.num_classes,), dtype=torch.float64, device="cuda")
            self._total_area_union = torch.zeros((self.num_classes,), dtype=torch.float64, device="cuda")
            self._total_area_pred = torch.zeros((self.num_classes,), dtype=torch.float64, device="cuda")
            self._total_area_label = torch.zeros((self.num_classes,), dtype=torch.float64, device="cuda")
        elif self._mode == "window_avg":
            self._total_area_intersect = TensorList(
                (self.window_size, self.num_classes), init_value=0, dtype=torch.float64, device="cuda", all_gather=False
            )
            self._total_area_union = TensorList(
                (self.window_size, self.num_classes), init_value=0, dtype=torch.float64, device="cuda", all_gather=False
            )
            self._total_area_pred = TensorList(
                (self.window_size, self.num_classes), init_value=0, dtype=torch.float64, device="cuda", all_gather=False
            )
            self._total_area_label = TensorList(
                (self.window_size, self.num_classes), init_value=0, dtype=torch.float64, device="cuda", all_gather=False
            )
            self._idx = torch.tensor([0], dtype=torch.int64, device="cuda")
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def reset(self):
        self._total_area_intersect.fill_(0)
        self._total_area_union.fill_(0)
        self._total_area_pred.fill_(0)
        self._total_area_label.fill_(0)

    def process(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        ignore_index: int = 255,
        label_map: Dict = {},
        reduce_zero_label: bool = False,
    ):
        area_intersect, area_union, area_pred, area_label = intersect_and_union(
            pred, label, self.num_classes, ignore_index, label_map, reduce_zero_label
        )

        if self._mode == "onetime":
            self._total_area_intersect += area_intersect
            self._total_area_union += area_union
            self._total_area_pred += area_pred
            self._total_area_label += area_label
        elif self._mode == "window_avg":
            self._total_area_intersect[self._idx] = area_intersect.unsqueeze(0)
            self._total_area_union[self._idx] = area_union.unsqueeze(0)
            self._total_area_pred[self._idx] = area_pred.unsqueeze(0)
            self._total_area_label[self._idx] = area_label.unsqueeze(0)
            self._idx = (self._idx + 1) % self.window_size

    def _aggregate_distributed(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._distributed:
            dist_utils.synchronize()
            dist.all_reduce(tensor)
        return tensor

    def calculate(self) -> Dict[str, float]:
        if self._mode == "onetime":
            total_area_intersect = self._total_area_intersect.clone()
            total_area_union = self._total_area_union.clone()
            total_area_pred = self._total_area_pred.clone()
            total_area_label = self._total_area_label.clone()
        elif self._mode == "window_avg":
            total_area_intersect = self._total_area_intersect._tensor.clone()
            total_area_union = self._total_area_union._tensor.clone()
            total_area_pred = self._total_area_pred._tensor.clone()
            total_area_label = self._total_area_label._tensor.clone()

        total_area_intersect = self._aggregate_distributed(total_area_intersect)
        total_area_union = self._aggregate_distributed(total_area_union)
        total_area_pred = self._aggregate_distributed(total_area_pred)
        total_area_label = self._aggregate_distributed(total_area_label)

        if self._mode == "window_avg":
            total_area_intersect = total_area_intersect.sum(dim=0)
            total_area_union = total_area_union.sum(dim=0)
            total_area_pred = total_area_pred.sum(dim=0)
            total_area_label = total_area_label.sum(dim=0)

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        metrics = {"aAcc": all_acc}
        iou = total_area_intersect / total_area_union
        acc = total_area_intersect / total_area_label
        metrics["mIoU"] = iou.mean()
        metrics["mAcc"] = acc.mean()

        return metrics

    def evaluate(self):
        metrics = self.calculate()
        logger.info(
            "Evaluation results: \n{}".format(
                tabulate([[metric, "{:.4f}".format(value)] for metric, value in metrics.items()], tablefmt="github",)
            )
        )
        return metrics

# Example usage
if __name__ == "__main__":
    # Define model
    model = build_optimized_model()

    # Load model weights, preprocess data, and evaluate
    # Assuming TensorFlow/Keras code for training and evaluation
    # TensorFlow/Keras code for predictions and processing
    # Convert TensorFlow/Keras predictions to PyTorch tensors if needed
    # Example:
    # predictions = model.predict(input_data)
    # predictions_tensor = torch.tensor(predictions).to("cuda")

    # Initialize evaluator
    evaluator = SegmentationEvaluator(num_classes=1, distributed=False, mode="onetime")

    # Process predictions
    # evaluator.process(predictions_tensor, labels_tensor)

    # Evaluate and log results
    metrics = evaluator.evaluate()
