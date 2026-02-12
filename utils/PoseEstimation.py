import numpy as np

def arun(A, B):
    """Solve 3D registration using Arun's method: B = RA + t
    retrieved (A) -> query (B)
    R and t are expressed in the query (B / cam3) frame, because the output points live in B
    """
    N = A.shape[1]
    assert B.shape[1] == N

    # calculate centroids
    A_centroid = np.reshape(1/N * (np.sum(A, axis=1)), (3,1))
    B_centroid = np.reshape(1/N * (np.sum(B, axis=1)), (3,1))

    # calculate the vectors from centroids
    A_prime = A - A_centroid
    B_prime = B - B_centroid

    # rotation estimation
    H = np.zeros([3, 3])
    for i in range(N):
        ai = A_prime[:, i]
        bi = B_prime[:, i]
        H = H + np.outer(ai, bi)
    U, S, V_transpose = np.linalg.svd(H)
    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)
    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose

    # translation estimation
    t = B_centroid - R @ A_centroid

    return R, t

def weighted_umeyama(A, B, w=None, with_scale=False, eps=1e-9):
    """
    Weighted Umeyama alignment: B ≈ s * R @ A + t

    Args:
        A, B : (3, N) arrays of corresponding 3D points
        w    : (N,) weights (larger = more trusted). If None → uniform.
        with_scale : estimate isotropic scale if True
        eps  : numerical stability

    Returns:
        R : (3,3) rotation
        t : (3,1) translation
        s : scale (only if with_scale=True)
    """
    assert A.shape[0] == 3 and B.shape[0] == 3
    assert A.shape[1] == B.shape[1]
    N = A.shape[1]

    if w is None:
        w = np.ones(N)
    w = w.reshape(1, -1)

    # normalize weights
    w_sum = np.sum(w) + eps
    w = w / w_sum

    # weighted centroids
    mu_A = A @ w.T        # (3,1)
    mu_B = B @ w.T        # (3,1)

    # centered points
    A_c = A - mu_A
    B_c = B - mu_B

    # weighted covariance
    H = A_c @ np.diagflat(w) @ B_c.T

    # SVD
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # reflection handling
    D = np.eye(3)
    if np.linalg.det(V @ U.T) < 0:
        D[-1, -1] = -1

    R = V @ D @ U.T

    # scale
    if with_scale:
        var_A = np.sum(w * np.sum(A_c**2, axis=0))
        s = np.trace(np.diag(S) @ D) / (var_A + eps)
    else:
        s = 1.0

    # translation
    t = mu_B - s * R @ mu_A

    if with_scale:
        return R, t, s
    else:
        return R, t