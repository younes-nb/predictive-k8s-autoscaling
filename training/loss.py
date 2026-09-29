import torch
import torch.nn as nn


def per_target_huber_loss(
    preds,
    target,
    cpu_beta: float = 0.01,
    mem_beta: float = 0.002,
    lambda_cpu: float = 0.5,
    lambda_mem: float = 0.5,
    rel_w: float = 0.0,
    rel_eps: float = 1e-6,
):
    t = preds.shape[-1]
    if t == 1:
        return nn.functional.mse_loss(preds, target)
    cpu_loss = nn.functional.smooth_l1_loss(
        preds[..., 0], target[..., 0], beta=cpu_beta
    )
    p_mem = preds[..., 1]
    t_mem = target[..., 1]
    mem_loss = nn.functional.smooth_l1_loss(p_mem, t_mem, beta=mem_beta)
    if rel_w:
        rel = torch.abs(p_mem - t_mem) / (torch.abs(t_mem) + rel_eps)
        mem_loss = mem_loss + rel_w * rel.mean()
    return lambda_cpu * cpu_loss + lambda_mem * mem_loss


def per_target_loss(preds, target, mem_mode="mse"):
    t = preds.shape[-1]
    if t == 1:
        return nn.functional.mse_loss(preds, target)
    cpu_loss = nn.functional.mse_loss(preds[..., 0], target[..., 0])
    if mem_mode == "l1":
        mem_loss = nn.functional.l1_loss(preds[..., 1], target[..., 1])
    else:
        mem_loss = nn.functional.mse_loss(preds[..., 1], target[..., 1])
    return cpu_loss + mem_loss


def asymmetric_huber_loss(
    preds,
    target,
    cpu_beta: float = 0.01,
    mem_beta: float = 0.002,
    lambda_cpu: float = 0.5,
    lambda_mem: float = 0.5,
    under_weight_cpu: float = 1.0,
    under_weight_mem: float = 1.0,
    rel_w: float = 0.0,
    rel_eps: float = 1e-6,
):
    t = preds.shape[-1]
    if t == 1:
        return nn.functional.smooth_l1_loss(preds, target, beta=cpu_beta)

    cpu_pred = preds[..., 0]
    cpu_target = target[..., 0]
    cpu_error = cpu_target - cpu_pred
    cpu_huber = nn.functional.smooth_l1_loss(cpu_pred, cpu_target, beta=cpu_beta, reduction='none')
    cpu_weight = torch.where(cpu_error > 0, under_weight_cpu, 1.0)
    cpu_loss = (cpu_huber * cpu_weight).mean()

    p_mem = preds[..., 1]
    t_mem = target[..., 1]
    mem_error = t_mem - p_mem
    mem_huber = nn.functional.smooth_l1_loss(p_mem, t_mem, beta=mem_beta, reduction='none')
    mem_weight = torch.where(mem_error > 0, under_weight_mem, 1.0)
    mem_loss = (mem_huber * mem_weight).mean()

    if rel_w:
        rel = torch.abs(p_mem - t_mem) / (torch.abs(t_mem) + rel_eps)
        mem_loss = mem_loss + rel_w * rel.mean()

    return lambda_cpu * cpu_loss + lambda_mem * mem_loss

