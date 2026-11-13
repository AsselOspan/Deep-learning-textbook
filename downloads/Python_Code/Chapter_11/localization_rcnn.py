"""Рамочная регрессия, NMS, AP и микроопыт общего прохода свёрток."""

import argparse
import math
from pathlib import Path
import sys
from time import perf_counter_ns

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Chapter_10"))
from classification_localization import box_iou


def to_center_size(boxes):
    lo, hi = boxes[..., :2], boxes[..., 2:]
    return (lo + hi) / 2, hi - lo


def encode_boxes(proposals, targets):
    center, size = to_center_size(proposals)
    target_center, target_size = to_center_size(targets)
    if (size <= 0).any() or (target_size <= 0).any():
        raise ValueError("Ширина и высота рамок должны быть положительными")
    return torch.cat(((target_center - center) / size, (target_size / size).log()), dim=-1)


def decode_boxes(proposals, delta):
    center, size = to_center_size(proposals)
    center = center + size * delta[..., :2]
    half = size * delta[..., 2:].exp() / 2
    return torch.cat((center - half, center + half), dim=-1)


def greedy_nms(boxes, scores, labels=None, threshold=0.5):
    """Подавляет IoU > threshold, при наличии labels только внутри класса."""
    order = sorted(range(len(boxes)), key=lambda i: (-float(scores[i]), i))
    keep = []
    while order:
        index, *remaining = order
        keep.append(index)
        order = [
            other for other in remaining
            if (labels is not None and labels[index] != labels[other])
            or box_iou(boxes[index], boxes[other]).item() <= threshold
        ]
    return keep


def average_precision(matches, total_targets):
    if total_targets <= 0 or not matches or sum(matches) > total_targets:
        raise ValueError("Нужен хотя бы один объект и непротиворечивый список TP/FP")
    running = 0
    precision, recall = [], []
    for rank, matched in enumerate(matches, 1):
        running += int(matched)
        precision.append(running / rank)
        recall.append(running / total_targets)
    envelope = [max(precision[i:]) for i in range(len(precision))]
    area = sum((current - previous) * p
               for previous, current, p in zip([0.0] + recall, recall, envelope))
    sampled = [max((p for p, r in zip(precision, recall) if r >= step / 10), default=0.0)
               for step in range(11)]
    return {"precision": precision, "recall": recall, "ap_area": area, "ap_voc07": sum(sampled) / 11}


def numeric_examples():
    proposals = torch.tensor([[10., 20., 30., 40.]], dtype=torch.float64)
    targets = torch.tensor([[14., 17., 38., 47.]], dtype=torch.float64)
    delta = encode_boxes(proposals, targets)
    torch.testing.assert_close(decode_boxes(proposals, delta), targets)
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.],
                          [12., 0., 22., 10.], [0., 0., 10., 10.]])
    scores = torch.tensor([.92, .86, .73, .68])
    labels = torch.tensor([0, 0, 0, 1])
    per_class = greedy_nms(boxes, scores, labels)
    agnostic = greedy_nms(boxes, scores)
    assert per_class == [0, 2, 3] and agnostic == [0, 2]
    ap = average_precision([True, False, True], 2)
    print("delta =", [round(x, 6) for x in delta[0].tolist()],
          "decoded =", decode_boxes(proposals, delta)[0].tolist())
    print(f"IoU(A,B)={box_iou(boxes[0], boxes[1]):.6f} "
          f"NMS classwise={per_class} class-agnostic={agnostic}")
    print(f"AP integral={ap['ap_area']:.6f} AP VOC07={ap['ap_voc07']:.6f}")
    return {"delta": delta, "boxes": boxes, "scores": scores, "labels": labels,
            "per_class": per_class, "agnostic": agnostic, "ap": ap}


class TinyRoutes:
    def __init__(self, seed, count):
        torch.manual_seed(seed)
        self.image = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(seed))
        self.boxes = [(x, y, x + 24, y + 24) for y in (4, 32, 60, 88) for x in (4, 32, 60, 88)]
        self.count = count
        self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(8, 16, 3, padding=1), nn.ReLU()).eval()
        self.head = nn.Linear(16 * 3 * 3, 6).eval()
        self.objectness = nn.Conv2d(16, 1, 1).eval()

    def head_on_regions(self, features, regions):
        pooled = [F.adaptive_max_pool2d(features[..., y0:y1, x0:x1], (3, 3))
                  for x0, y0, x1, y1 in regions]
        return self.head(torch.cat(pooled).flatten(1))

    def crop_first(self):
        patches = []
        for x0, y0, x1, y1 in self.boxes[:self.count]:
            crop = self.image[..., y0:y1, x0:x1]
            resized = F.interpolate(crop, size=(64, 64), mode="bilinear", align_corners=False)
            features = self.backbone(resized)
            patches.append(F.adaptive_max_pool2d(features, (3, 3)))
        return self.head(torch.cat(patches).flatten(1))

    def shared_external(self):
        return self.head_on_regions(self.backbone(self.image), self.boxes[:self.count])

    def shared_score(self):
        features = self.backbone(self.image)
        score = self.objectness(features)[0, 0]
        values = [float(score[(y0+y1)//2, (x0+x1)//2]) for x0, y0, x1, y1 in self.boxes]
        ids = sorted(range(len(values)), key=lambda i: (-values[i], i))[:self.count]
        return self.head_on_regions(features, [self.boxes[i] for i in ids])


@torch.inference_mode()
def measure(seed, proposals, repeats):
    routes = TinyRoutes(seed, proposals)
    paths = {"crop_first": routes.crop_first, "shared_external": routes.shared_external,
             "shared_score": routes.shared_score}
    values = {name: [] for name in paths}
    order = list(paths)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(3):
            for call in paths.values():
                assert call().shape == (proposals, 6)
        for step in range(repeats):
            offset = step % len(order)
            for name in order[offset:] + order[:offset]:
                start = perf_counter_ns()
                paths[name]()
                values[name].append((perf_counter_ns() - start) / 1e6)
    finally:
        torch.set_num_threads(previous)
    result = {}
    for name, times in values.items():
        p50, p95 = torch.quantile(torch.tensor(times), torch.tensor([.5, .95])).tolist()
        passes = proposals if name == "crop_first" else 1
        sites = proposals * 64 * 64 if name == "crop_first" else 128 * 128
        result[name] = dict(p50_ms=p50, p95_ms=p95, backbone_passes=passes, sites=sites)
        print(f"{name}: passes={passes} sites={sites} "
              f"p50_ms={p50:.3f} p95_ms={p95:.3f}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("short", "full"), default="full")
    parser.add_argument("--proposals", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    defaults = {"short": (6, 9), "full": (12, 30)}
    proposals, repeats = defaults[args.profile]
    if args.proposals is not None:
        proposals = args.proposals
    if not 1 <= proposals <= 16:
        parser.error("Число регионов должно быть от 1 до 16")
    print(f"profile={args.profile} torch={torch.__version__} seed={args.seed} "
          f"regions={proposals} repeats={repeats} CPU_threads_for_timing=1")
    numbers = numeric_examples()
    result = measure(args.seed, proposals, repeats)
    return {"numerical": numbers, "routes": result, "proposals": proposals, "repeats": repeats}


if __name__ == "__main__":
    main()
