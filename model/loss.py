import torch


def soft_dice_loss(outputs, targets, per_image=False, eps=1e-5):
    batch_size = outputs.size()[0]
    if not per_image:
        batch_size = 1
    dice_target = targets.reshape(batch_size, -1).float()
    dice_output = outputs.reshape(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def focal_cannab(outputs, targets, gamma=2, ignore_index=255, eps=1e-8):
    non_ignored = targets.reshape(-1) != ignore_index
    targets = targets.reshape(-1)[non_ignored].float()
    outputs = outputs.reshape(-1)[non_ignored]
    outputs = torch.clamp(outputs, eps, 1 - eps)
    targets = torch.clamp(targets, eps, 1 - eps)
    pt = (1 - targets) * (1 - outputs) + targets * outputs
    return (-(1. - pt) ** gamma * torch.log(pt)).mean()
