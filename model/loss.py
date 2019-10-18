import torch


def soft_dice_loss(outputs, targets, eps=1e-5):
    batch_size = outputs.shape[0]
    dice_target = targets.reshape(batch_size, -1).float()
    dice_output = outputs.reshape(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union)
    return loss


def focal_cannab(outputs, targets, gamma=2, eps=1e-8):
    batch_size = outputs.shape[0]
    targets = targets.reshape(batch_size, -1).float()
    outputs = outputs.reshape(batch_size, -1)
    outputs = torch.clamp(outputs, eps, 1 - eps)
    targets = torch.clamp(targets, eps, 1 - eps)
    pt = (1 - targets) * (1 - outputs) + targets * outputs
    y = (1 - pt) ** gamma
    return (-y / y.mean(dim=-1).unsqueeze(-1) * torch.log(pt)).mean(dim=-1)
