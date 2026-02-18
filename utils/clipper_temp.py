import numpy as np
from dataclasses import dataclass

SQRT_TWO_THIRDS = np.sqrt(2.0 / 3.0)
SQRT_ONE_THIRD = np.sqrt(1.0 / 3.0)

@dataclass
class Params:
    point_dim: int
    mindist: float
    epsilon: float
    sigma: float
    drift_aware: bool
    drift_scale: float
    drift_scale_sigma: bool
    gravity_guided: bool
    gravity_unc_ang_rad: float

def geometric_mean(x):
    """
    x: 1D array-like of positive numbers
    """
    x = np.asarray(x)
    return np.prod(x) ** (1.0 / len(x))

def fuse_matrix_geometric(M, scores, eps=1e-12):
    """
    M: (m, m) distance consistency matrix
    scores_D: (m,) association confidences
    """

    M = np.asarray(M)
    scores = np.asarray(scores)

    # Avoid log(0) or zero products
    M = np.clip(M, eps, None)
    scores = np.clip(scores, eps, None)

    # Create broadcastable row/column confidence matrices
    ci = scores[:, None]   # shape (m, 1)
    cj = scores[None, :]   # shape (1, m)

    # Geometric mean
    W = (M * ci * cj) ** (1.0 / 3.0)

    return W

def fuse_matrix_weighted(M, scores_D, alpha=1.0, beta=1.0, eps=1e-12):

    M = np.clip(M, eps, None)
    scores_D = np.clip(scores_D, eps, None)

    log_M = np.log(M)
    log_ci = np.log(scores_D)[:, None]
    log_cj = np.log(scores_D)[None, :]

    log_W = (
        alpha * log_M +
        beta * log_ci +
        beta * log_cj
    ) / (alpha + 2 * beta)

    return np.exp(log_W)

def build_C_from_M(M, eps=1e-12):
    return (np.abs(M) > eps).astype(np.float64)

# Concatenate features of each cluster
# |---- point_dim ----|
#                      |---- ratio_feature_dim ----|
#                                                  |---- cos_feature_dim ----|
def cluster_info_to_clipper_list(centroid, ratio_feat = None, descriptor = None):        
    # Centroid (to be compared using distance metric)
    cluster_info_as_list = centroid.reshape(-1).tolist()[:3]

    # Some scalar comparison
    if ratio_feat is not None:
        cluster_info_as_list += [ratio_feat]

    # Learned descriptor (to be compared using cosine similarity)
    if descriptor is not None:
        cluster_info_as_list += np.array(descriptor).tolist()
    return cluster_info_as_list 
    
# The pairwise consistency score for the association of points (ai,bi) and (aj,bj)
def pairwise_similarity(ai, aj, bi, bj, params,
                        gravity_unc_ang_cos=1.0,
                        gravity_unc_ang_sin=0.0):
    """
    ai, aj, bi, bj: numpy arrays
    params: object or dict with required fields
    """

    # distance between two points in same cloud
    l1 = np.linalg.norm(ai[:params.point_dim] - aj[:params.point_dim])
    l2 = np.linalg.norm(bi[:params.point_dim] - bj[:params.point_dim])

    # minimum distance criterion
    if params.mindist > 0 and (l1 < params.mindist or l2 < params.mindist):
        return 0.0

    # drift-aware epsilon
    if params.drift_aware:
        epsilon = max(
            params.epsilon,
            params.epsilon * params.drift_scale * 0.5 * (l1 + l2)
        )
    else:
        epsilon = params.epsilon

    if params.drift_aware and params.drift_scale_sigma:
        sigma = max(
            params.sigma,
            params.sigma * params.drift_scale * 0.5 * (l1 + l2)
        )
    else:
        sigma = params.sigma

    # gravity-guided mode
    if params.gravity_guided:

        xy_dist1 = np.linalg.norm(ai[:2] - aj[:2])
        xy_dist2 = np.linalg.norm(bi[:2] - bj[:2])

        z_diff1 = ai[2] - aj[2]
        z_diff2 = bi[2] - bj[2]

        c_xy = abs(xy_dist1 - xy_dist2)
        c_z = abs(z_diff1 - z_diff2)

        sigma_xy = sigma
        sigma_z = sigma
        epsilon_xy = epsilon
        epsilon_z = epsilon

        if params.gravity_unc_ang_rad > 0.0:
            xy_mean = 0.5 * (xy_dist1 + xy_dist2)
            z_mean = 0.5 * (abs(z_diff1) + abs(z_diff2))

            sigma_xy += abs(xy_mean * gravity_unc_ang_cos - xy_mean)
            sigma_z += abs(z_mean * gravity_unc_ang_sin)
            epsilon_xy += abs(xy_mean * gravity_unc_ang_cos - xy_mean)
            epsilon_z += abs(z_mean * gravity_unc_ang_sin)

        if (c_xy > SQRT_TWO_THIRDS * epsilon_xy or
            c_z > SQRT_ONE_THIRD * epsilon_z):
            return 0.0

        return np.exp(
            -0.5 * (
                c_xy**2 / (2.0/3.0 * sigma_xy**2) +
                c_z**2 / (sigma_z**2 / 3.0)
            )
        )

    else:
        # standard distance similarity
        c = abs(l1 - l2)
        if c > epsilon:
            return 0.0
        return np.exp(-0.5 * c*c / (sigma*sigma))

# The consistency score for the fused pairwise and single scores (using geometric mean)
def pairwise_single_fusion(pair_ij, single_i, single_j, params):
    """
    Geometric mean fusion only.

    pair_ij: pairwise similarity
    single_i: single similarity for i
    single_j: single similarity for j
    params.distance_weight: weight on pairwise term
    params.ratio_feature_dim
    params.cos_feature_dim
    """

    # If no feature similarity is used, just return pair score
    if params.ratio_feature_dim <= 0 and params.cos_feature_dim <= 0:
        return pair_ij

    w = params.distance_weight

    # Equivalent to:
    # (pair_ij^w * single_i * single_j)^(1/(w+2))
    return (pair_ij**w * single_i * single_j) ** (1.0 / (w + 2.0))

# The consistency score for the single association of (ai,bi)
def single_similarity(ai, bi, params):
    """
    ai, bi: 1D numpy arrays (Datum)
    params: object with required attributes
    """

    # No feature similarity used
    if params.cos_feature_dim == 0 and params.ratio_feature_dim == 0:
        return 1.0

    cosine_score_scaled = 0.0
    ratio_score = 0.0

    # -------------------------
    # Cosine feature similarity
    # -------------------------
    if params.cos_feature_dim > 0:
        start = params.point_dim + params.ratio_feature_dim
        end = start + params.cos_feature_dim

        ai_feat = ai[start:end]
        bi_feat = bi[start:end]

        denom = np.linalg.norm(ai_feat) * np.linalg.norm(bi_feat)
        if denom == 0:
            return 0.0

        cosine_score = np.dot(ai_feat, bi_feat) / denom

        if cosine_score >= params.cosine_max:
            cosine_score_scaled = 1.0
        elif cosine_score <= params.cosine_min:
            return 0.0  # early exit
        else:
            cosine_score_scaled = (
                (cosine_score - params.cosine_min) /
                (params.cosine_max - params.cosine_min)
            )

        if params.ratio_feature_dim == 0:
            return cosine_score_scaled

    # -------------------------
    # Ratio feature similarity
    # -------------------------
    if params.ratio_feature_dim > 0:
        start = params.point_dim
        end = start + params.ratio_feature_dim

        ai_ratio = ai[start:end]
        bi_ratio = bi[start:end]

        # similarity = min / max per dimension
        mins = np.minimum(ai_ratio, bi_ratio)
        maxs = np.maximum(ai_ratio, bi_ratio)

        # Avoid division by zero
        if np.any(maxs == 0):
            return 0.0

        ratio_scores = mins / maxs

        # Threshold check
        if np.any(ratio_scores < params.ratio_epsilon):
            return 0.0

        # Geometric mean of ratio scores
        ratio_score = np.prod(ratio_scores) ** (1.0 / params.ratio_feature_dim)

        if params.cos_feature_dim == 0:
            return ratio_score

    # -------------------------
    # Fuse cosine + ratio
    # -------------------------
    return (
        (ratio_score ** params.ratio_weight) *
        (cosine_score_scaled ** params.cosine_weight)
    ) ** (1.0 / (params.ratio_weight + params.cosine_weight))