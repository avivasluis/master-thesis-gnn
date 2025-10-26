import torch

def compute_density(n_nodes, n_edges):
    n_max_edges = (n_nodes*(n_nodes-1))/2
    density = (n_edges / n_max_edges) * 100
    return density

def compute_homophily_ratio(edge_index: torch.Tensor, y: torch.Tensor):
    src, dst = edge_index
    mask = src != dst
    same = (y[src[mask]] == y[dst[mask]]).float().mean().item()
    return same

def compute_assortativity_categorical(edge_index: torch.Tensor, y: torch.Tensor, num_classes: int = None):
    # Treat as undirected by counting both directions, per Newman (2003)
    if num_classes is None:
        num_classes = int(y.max().item()) + 1

    src, dst = edge_index
    # Remove self-loops (optional but typical for this metric)
    mask = src != dst
    src, dst = src[mask], dst[mask]

    # Make edges undirected by adding both directions
    src2 = torch.cat([src, dst], dim=0)
    dst2 = torch.cat([dst, src], dim=0)

    pairs = torch.stack([y[src2], y[dst2]], dim=1)  # [2M, 2] class pairs
    idx = pairs[:, 0] * num_classes + pairs[:, 1]
    counts = torch.bincount(idx, minlength=num_classes * num_classes).double()
    e = counts.view(num_classes, num_classes)
    e = e / e.sum()  # mixing matrix with sum=1 over all (i,j)

    a = e.sum(dim=1)  # row sums
    b = e.sum(dim=0)  # col sums (equal to a in undirected case)
    expected = (a * b).sum()
    r = (e.diag().sum() - expected) / (1 - expected + 1e-12)
    return r.item()