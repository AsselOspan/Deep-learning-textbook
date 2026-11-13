"""Классификация, локализация и сегментация фигур на синтетических кадрах."""

import argparse
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


SIDE = 32
PROFILES = {"short": (256, 64, 64, 3), "full": (800, 160, 160, 8)}


def box_corners(center_size):
    center, half = center_size[..., :2], center_size[..., 2:] / 2
    return torch.cat((center - half, center + half), dim=-1)


def box_iou(a, b):
    """Попарный IoU согласованных рамок xyxy с полуоткрытыми границами."""
    size = (torch.minimum(a[..., 2:], b[..., 2:]) -
            torch.maximum(a[..., :2], b[..., :2])).clamp_min(0)
    intersection = size.prod(dim=-1)
    union = ((a[..., 2:] - a[..., :2]).clamp_min(0).prod(dim=-1)
             + (b[..., 2:] - b[..., :2]).clamp_min(0).prod(dim=-1) - intersection)
    return torch.where(union > 0, intersection / union.clamp_min(1e-8), 0.0)


def mask_iou(a, b):
    a, b = a.bool().flatten(1), b.bool().flatten(1)
    intersection, union = (a & b).sum(1), (a | b).sum(1)
    return intersection / union.clamp_min(1)


def mask_to_box(mask):
    rows, cols = mask.bool().nonzero(as_tuple=True)
    if rows.numel() == 0:
        raise ValueError("Пустая маска не определяет рамку")
    return torch.stack((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))


def make_examples(count, seed):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % 2
    size = 2 * torch.randint(4, 7, (count,), generator=generator)
    left = 2 + (torch.rand(count, generator=generator) * (SIDE - size - 3)).long()
    top = 2 + (torch.rand(count, generator=generator) * (SIDE - size - 3)).long()
    axis = torch.arange(SIDE)
    x, y = axis[None, None, :] - left[:, None, None], axis[None, :, None] - top[:, None, None]
    s = size[:, None, None]
    square = (x >= 0) & (x < s) & (y >= 0) & (y < s)
    circle = (x - (s - 1) / 2).square() + (y - (s - 1) / 2).square() <= (s / 2).square()
    masks = torch.where(labels[:, None, None] == 0, square, circle).float().unsqueeze(1)
    noise = torch.randn(masks.shape, generator=generator) * 0.10
    images = (0.20 + 0.65 * masks + noise).clamp(0, 1)
    boxes = torch.stack((left + size / 2, top + size / 2, size, size), dim=1) / SIDE
    return TensorDataset(images, labels, boxes, masks)


class ShapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shallow = nn.Sequential(nn.Conv2d(1, 12, 3, padding=1), nn.ReLU())
        self.deep = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(24 * 8 * 8, 64), nn.ReLU())
        self.category = nn.Linear(64, 2)
        self.box = nn.Linear(64, 4)
        self.mask = nn.Sequential(nn.Conv2d(36, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 1, 1))

    def forward(self, images):
        fine = self.shallow(images)
        coarse = self.deep(fine)
        hidden = self.shared(coarse)
        expanded = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        return self.category(hidden), self.box(hidden).sigmoid(), self.mask(torch.cat((fine, expanded), 1))


def objective(outputs, labels, boxes, masks):
    category, box, mask = outputs
    return (F.cross_entropy(category, labels)
            + 8 * F.smooth_l1_loss(box, boxes, beta=0.05)
            + 0.5 * F.binary_cross_entropy_with_logits(mask, masks))


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    total = torch.zeros(5)
    example = None
    for images, labels, boxes, masks in batches:
        outputs = model(images.to(device))
        category, predicted, mask_logits = (value.cpu() for value in outputs)
        match = category.argmax(1) == labels
        predicted_box = box_corners(predicted).clamp(0, 1)
        true_box = box_corners(boxes)
        overlap = box_iou(predicted_box, true_box)
        overlap_mask = mask_iou(mask_logits >= 0, masks)
        total += torch.tensor([objective((category, predicted, mask_logits), labels, boxes, masks) * len(images),
                               match.sum(), overlap.sum(), (match & (overlap >= 0.5)).sum(),
                               overlap_mask.sum()])
        if example is None:
            example = {"image": images[0, 0], "true_mask": masks[0, 0],
                       "predicted_mask": mask_logits[0, 0] >= 0,
                       "true_box": true_box[0] * SIDE, "predicted_box": predicted_box[0] * SIDE,
                       "true_label": labels[0].item(), "predicted_label": category[0].argmax().item()}
    values = (total / len(batches.dataset)).tolist()
    return dict(zip(("loss", "accuracy", "box_miou", "joint_at_05", "mask_miou"), values)), example


def show(stage, values):
    print(stage, " ".join(f"{name}={number:.4f}" for name, number in values.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args(argv)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    n_train, n_val, n_test, epochs = PROFILES[args.profile]
    train, val, test = (make_examples(n, args.seed + i)
                        for i, n in enumerate((n_train, n_val, n_test)))
    train_loader = DataLoader(train, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader, test_loader = DataLoader(val, batch_size=32), DataLoader(test, batch_size=32)
    model = ShapeNet().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    print(f"profile={args.profile} torch={torch.__version__} device={device} seed={args.seed} "
          f"train={n_train} val={n_val} test={n_test} parameters={parameters}")
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for images, labels, boxes, masks in train_loader:
            images, labels, boxes, masks = (v.to(device) for v in (images, labels, boxes, masks))
            optimizer.zero_grad()
            loss = objective(model(images), labels, boxes, masks)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(images)
        values, _ = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={running / n_train:.4f}")
        show("validation", values)
        if values["loss"] < best:
            best, best_epoch = values["loss"], epoch
            checkpoint = deepcopy(model.state_dict())
    model.load_state_dict(checkpoint)
    values, example = evaluate(model, test_loader, device)
    print(f"best_epoch={best_epoch}")
    show("test", values)
    print("example", example["true_box"].tolist(), example["predicted_box"].tolist())
    return dict(values, best_epoch=best_epoch, parameters=parameters, example=example)


if __name__ == "__main__":
    main()
