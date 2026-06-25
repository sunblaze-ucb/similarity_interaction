import torch


def cka(feats_A, feats_B, kernel_metric='ip', unbiased=True):
    """Centered Kernel Alignment (CKA) between two feature matrices.

    kernel_metric: 'ip' (linear) or 'rbf'
    unbiased: True for unbiased HSIC estimator, False for the biased estimator
    """
    A = (feats_A - feats_A.mean(dim=0, keepdim=True)).to(torch.float64)
    B = (feats_B - feats_B.mean(dim=0, keepdim=True)).to(torch.float64)

    if kernel_metric == 'ip':
        K = torch.mm(A, A.T)
        L = torch.mm(B, B.T)
    elif kernel_metric == 'rbf':
        dists = torch.pdist(A, p=2)
        rbf_sigma1 = torch.median(dists)
        K = torch.exp(-torch.cdist(A, A) ** 2 / (2 * rbf_sigma1 ** 2))

        dists = torch.pdist(B, p=2)
        rbf_sigma2 = torch.median(dists)
        L = torch.exp(-torch.cdist(B, B) ** 2 / (2 * rbf_sigma2 ** 2))
    else:
        raise ValueError(f"Invalid kernel metric {kernel_metric}")

    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    hsic_kk = hsic_fn(K, K)
    hsic_ll = hsic_fn(L, L)
    hsic_kl = hsic_fn(K, L)

    cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
    if (torch.isnan(cka_value) or cka_value < 0):
        print(f"[CKA-debug] hsic_kl = {hsic_kl.item():.4e}, "
              f"hsic_kk = {hsic_kk.item():.4e}, "
              f"hsic_ll = {hsic_ll.item():.4e}, "
              f"product = {(hsic_kk*hsic_ll).item():.4e}")

    return cka_value.item()


def hsic_unbiased(K, L):
    """Unbiased HSIC (Song et al., 2012, Eq. 5)."""
    m = K.shape[0]

    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """Biased HSIC (original CKA formulation)."""
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - torch.ones(n, n, dtype=K.dtype, device=K.device) / n
    return torch.trace(K @ H @ L @ H)
