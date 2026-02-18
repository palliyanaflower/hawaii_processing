import numpy as np
import cv2

import teaserpp_python
def teaser_solver(A, B, noise_bound=0.01):
    """
    A: (3, N) numpy array
    B: (3, N) numpy array
    Returns R, t such that B ≈ R @ A + t
    """

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 0.5
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(A, B)

    solution = solver.getSolution()

    return solution.rotation, solution.translation

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

def weighted_arun(A, B, w):
    """
    Weighted 3D registration using Arun's method:
        B = R A + t

    A: (3, N)
    B: (3, N)
    w: (N,)  non-negative weights

    Returns:
        R (3x3), t (3x1)
    """

    assert A.shape[0] == 3
    assert B.shape[0] == 3
    assert A.shape[1] == B.shape[1]
    assert A.shape[1] == w.shape[0]

    N = A.shape[1]

    # Ensure weights are column vector
    w = w.reshape(-1)

    # Normalize weights (not strictly required but improves conditioning)
    w_sum = np.sum(w)
    if w_sum <= 0:
        raise ValueError("Sum of weights must be positive.")
    w = w / w_sum

    # ----- Weighted centroids -----
    A_centroid = (A * w).sum(axis=1, keepdims=True)
    B_centroid = (B * w).sum(axis=1, keepdims=True)

    # ----- Centered coordinates -----
    A_prime = A - A_centroid
    B_prime = B - B_centroid

    # ----- Weighted cross-covariance -----
    H = np.zeros((3, 3))
    for i in range(N):
        ai = A_prime[:, i]
        bi = B_prime[:, i]
        H += w[i] * np.outer(ai, bi)

    # ----- SVD -----
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Reflection handling
    R = V @ np.diag([1, 1, np.sign(np.linalg.det(V @ U.T))]) @ U.T

    # ----- Translation -----
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
    





def skew(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])


def se3_exp(xi):
    """Exponential map from se(3) to SE(3)
    xi = [omega (3), upsilon (3)]
    """
    omega = xi[:3]
    upsilon = xi[3:]
    theta = np.linalg.norm(omega)

    if theta < 1e-8:
        R = np.eye(3)
        V = np.eye(3)
    else:
        omega_hat = skew(omega / theta)
        R = (
            np.eye(3)
            + np.sin(theta) * omega_hat
            + (1 - np.cos(theta)) * omega_hat @ omega_hat
        )
        V = (
            np.eye(3)
            + (1 - np.cos(theta)) / theta * omega_hat
            + (theta - np.sin(theta)) / theta * omega_hat @ omega_hat
        )

    t = V @ upsilon
    return R, t


def huber_weight(r, delta=1.0):
    norm = np.linalg.norm(r)
    if norm <= delta:
        return 1.0
    return delta / norm


def robust_se3_registration(A, B, max_iters=100):
    """
    Robust SE(3) Gauss–Newton refinement
    A, B: 3xN arrays
    """

    # ---- initialize with Arun
    R, t = arun(A, B)

    # Force correct shapes
    R = R.reshape(3, 3)
    t = t.reshape(3)

    for _ in range(max_iters):

        H = np.zeros((6, 6))
        b_vec = np.zeros(6)

        for i in range(A.shape[1]):

            a = A[:, i]        # (3,)
            b_i = B[:, i]      # (3,)

            p = R @ a + t      # (3,)
            r = p - b_i        # (3,)

            w = huber_weight(r)

            J = np.zeros((3, 6))
            J[:, :3] = -R @ skew(a)
            J[:, 3:] = np.eye(3)

            H += w * (J.T @ J)
            b_vec += w * (J.T @ r)

        dx = -np.linalg.solve(H, b_vec)

        dx = dx.flatten()  # ensure (6,)

        dR, dt = se3_exp(dx)

        # Enforce correct shapes again
        dR = dR.reshape(3, 3)
        dt = dt.reshape(3)

        R = dR @ R
        t = dR @ t + dt

        if np.linalg.norm(dx) < 1e-6:
            break

    return R, t

def se3_with_vertical_penalty(
    A,
    B,
    weights=None,
    lambda_z=1000.0,
    return_diagnostics=False
):
    """
    Full SE(3) solve using SVD (Arun) with vertical translation penalty.

    Args:
        A, B: 3xN numpy arrays (corresponding 3D points)
        weights: optional length-N weights
        lambda_z: penalty weight for vertical translation
                  (large -> suppress tz)
        return_diagnostics: if True, return conditioning info

    Returns:
        R: 3x3 rotation
        t: 3x1 translation
        (optional) diagnostics dict
    """

    assert A.shape[0] == 3 and B.shape[0] == 3
    N = A.shape[1]
    assert B.shape[1] == N

    if weights is None:
        weights = np.ones(N)

    weights = weights / np.sum(weights)

    # --------------------------------------------------
    # 1️⃣ Weighted centroids
    # --------------------------------------------------
    centroid_A = np.sum(A * weights, axis=1, keepdims=True)
    centroid_B = np.sum(B * weights, axis=1, keepdims=True)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # --------------------------------------------------
    # 2️⃣ Weighted covariance matrix
    # --------------------------------------------------
    H = (A_centered * weights) @ B_centered.T

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # --------------------------------------------------
    # 3️⃣ Translation solve
    # --------------------------------------------------
    RA_bar = R @ centroid_A
    t = centroid_B - RA_bar

    # --------------------------------------------------
    # 4️⃣ Vertical penalty
    # --------------------------------------------------
    # Equivalent to minimizing:
    # sum ||B - (RA + t)||^2 + lambda_z * t_z^2

    t[2, 0] /= (1.0 + lambda_z)

    if not return_diagnostics:
        return R, t

    diagnostics = {
        "singular_values_3d": S,
        "conditioning_ratio_3d": S[2] / S[0] if S[0] > 1e-12 else 0.0,
        "tz_before_penalty": float((centroid_B - RA_bar)[2, 0]),
        "tz_after_penalty": float(t[2, 0])
    }

    return R, t, diagnostics


def estimate_pose_from_2d2d(
    kpts1, kpts2,
    K1, dist1,
    K2, dist2
):
    """
    Estimate relative pose between two cameras with distortion.

    kpts1, kpts2 : Nx2 pixel coordinates (raw distorted)
    K1, K2       : 3x3 intrinsics
    dist1, dist2 : distortion coefficients

    Returns:
        R         : 3x3 rotation matrix (cam1 -> cam2, R in cam2 frame)
        t_unit    : 3-vector unit translation direction
        mask_pose : inlier mask
    """

    # --------------------------------------------------
    # 1️⃣ Undistort AND normalize to pinhole coordinates
    # --------------------------------------------------

    kpts1_norm = cv2.undistortPoints(
        kpts1.reshape(-1, 1, 2),
        K1,
        dist1
    ).reshape(-1, 2)

    kpts2_norm = cv2.undistortPoints(
        kpts2.reshape(-1, 1, 2),
        K2,
        dist2
    ).reshape(-1, 2)

    # --------------------------------------------------
    # 2️⃣ Estimate Essential matrix in normalized space
    # --------------------------------------------------

    E, mask = cv2.findEssentialMat(
        kpts1_norm,
        kpts2_norm,
        focal=1.0,
        pp=(0., 0.),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1e-3  # IMPORTANT: much smaller than pixel threshold
    )

    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")

    # --------------------------------------------------
    # 3️⃣ Recover pose (already normalized)
    # --------------------------------------------------

    _, R, t_unit, mask_pose = cv2.recoverPose(
        E,
        kpts1_norm,
        kpts2_norm
    )

    return R, t_unit.squeeze(), mask_pose.squeeze()
