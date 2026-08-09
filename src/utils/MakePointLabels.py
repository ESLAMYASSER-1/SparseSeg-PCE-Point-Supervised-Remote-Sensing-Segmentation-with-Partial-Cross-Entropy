import torch

def make_point_labels(full_mask, fraction, ignore_index=-1):
    device = full_mask.device
    g = torch.Generator(device=device)

    points = torch.full_like(full_mask, ignore_index)
    B, H, W = full_mask.shape

    for b in range(B):
        valid = torch.ones(H, W, dtype=torch.bool, device=device)
        coords = torch.nonzero(valid, as_tuple=False)
        n = max(1, int(round(coords.shape[0] * fraction)))
        idx = torch.randperm(coords.shape[0], generator=g, device=device)[:n]
        selected = coords[idx]
        points[b, selected[:, 0], selected[:, 1]] = full_mask[b, selected[:, 0], selected[:, 1]]
    return points