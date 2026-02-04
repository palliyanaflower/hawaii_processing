import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import numpy as np

def show_lidar_2d(
    cameras,
    figsize=(18, 8),
    kp_size=5,
    cmap="turbo",
):

    fig, axes = plt.subplots(
        1,
        len(cameras),
        figsize=figsize,
        sharex=False,
        sharey=False,
    )

    if len(cameras) == 1:
        axes = [axes]

    for ax, cam in zip(axes, cameras):
        ax.imshow(cam.img)

        # Matched keypoints
        ax.scatter(
            cam.u,
            cam.v,
            s=kp_size,
            c="gray",
            label="lidar points",
        )

        ax.set_title(f"{cam.name} – 2D projection")
        ax.axis("off")

    axes[0].legend(loc="lower right")
    plt.tight_layout()

    return fig, axes

def show_points_2d(
    cameras,
    keypoints,
    pc_clustered,
    figsize=(18, 8),
    kp_size=25,
    cmap="turbo",
):

    fig, axes = plt.subplots(
        1,
        len(cameras),
        figsize=figsize,
        sharex=False,
        sharey=False,
    )

    if len(cameras) == 1:
        axes = [axes]

    for ax, cam in zip(axes, cameras):
        ax.imshow(cam.img)

        # Matched keypoints
        ax.scatter(
            keypoints[cam.name][:, 0],
            keypoints[cam.name][:, 1],
            s=kp_size,
            c="red",
            label="Matched keypoints",
        )

        # Cluster centroids
        ax.scatter(
            pc_clustered[cam.name].pxs[:, 0],
            pc_clustered[cam.name].pxs[:, 1],
            s=kp_size,
            # c="cyan",
            c=pc_clustered[cam.name].kidxs,
            cmap=cmap,
            label="Cluster centroids",
        )

        ax.set_title(f"{cam.name} – 2D projection")
        ax.axis("off")

    axes[0].legend(loc="lower right")
    plt.tight_layout()

    return fig, axes

def show_points_3d(
    cameras,
    pc_clustered,
    figsize=(18, 8),
    all_lidar_size=1,
    centroid_size=5,
    cmap="turbo",
    elev=20,
    azim=-70,
):

    all_kidxs = np.concatenate([
        pc_clustered[cam.name].kidxs
        for cam in cameras
    ])

    # Optional: ignore invalid entries
    all_kidxs = all_kidxs[all_kidxs >= 0]

    norm = plt.Normalize(
        vmin=all_kidxs.min(),
        vmax=all_kidxs.max(),
    )

    fig = plt.figure(figsize=figsize)

    for i, cam in enumerate(cameras):

        ax = fig.add_subplot(
            1, len(cameras), i + 1, projection="3d"
        )

        # All LiDAR points
        ax.scatter(
            cam.pts_infront[:, 0],
            cam.pts_infront[:, 1],
            cam.pts_infront[:, 2],
            s=all_lidar_size,
            c="gray",
            alpha=0.4,
            label="All LiDAR",
        )

        # Cluster centroids
        valid = pc_clustered[cam.name].kidxs >= 0

        ax.scatter(
            pc_clustered[cam.name].pts[valid, 0],
            pc_clustered[cam.name].pts[valid, 1],
            pc_clustered[cam.name].pts[valid, 2],
            s=centroid_size,
            c=pc_clustered[cam.name].kidxs[valid],
            cmap=cmap,
            norm=norm,
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    return fig


def show_lidar_neighbors_2d(
    cameras,
    keypoints,
    nn,
    figsize=(18, 8),
    kp_size=25,
    lidar_size=1,
    cmap="turbo",
):
    """
    Visualize matched keypoints and their LiDAR neighbors in image space.

    Parameters
    ----------
    cameras : list
        List of camera objects. Each must have:
        - cam.name
        - cam.img
    keypoints : dict
        keypoints[cam.name] -> (N, 2) array of pixel coordinates
    nn : dict
        nn[cam.name] must have:
        - pxs_all      : (M, 2) LiDAR pixel projections
        - group_ids   : (M,) group / keypoint ids
    """
    fig, axes = plt.subplots(
        1,
        len(cameras),
        figsize=figsize,
        sharex=False,
        sharey=False,
    )

    if len(cameras) == 1:
        axes = [axes]

    all_group_ids = np.concatenate([
        nn[cam.name].group_ids for cam in cameras
    ])

    norm = plt.Normalize(
        vmin=all_group_ids.min(),
        vmax=all_group_ids.max(),
    )

    for ax, cam in zip(axes, cameras):
        ax.imshow(cam.img)

        # Matched keypoints
        ax.scatter(
            keypoints[cam.name][:, 0],
            keypoints[cam.name][:, 1],
            s=kp_size,
            c="red",
            label="Matched keypoints",
        )

        # LiDAR neighbors
        ax.scatter(
            nn[cam.name].pxs_all[:, 0],
            nn[cam.name].pxs_all[:, 1],
            s=lidar_size,
            c=nn[cam.name].group_ids,
            cmap=cmap,
            norm=norm,
            alpha=0.7,
            label="LiDAR neighbors",
        )


        ax.set_title(f"{cam.name} – 2D projection")
        ax.axis("off")

    axes[0].legend(loc="lower right")
    plt.tight_layout()

    return fig, axes


def show_lidar_neighbors_3d(
    cameras,
    nn,
    figsize=(18, 8),
    all_lidar_size=1,
    neighbor_size=5,
    cmap="turbo",
    elev=20,
    azim=-70,
):
    """
    Visualize LiDAR points and keypoint-associated neighbors in 3D.

    Parameters
    ----------
    cameras : list
        List of camera objects. Each must have:
        - cam.name
        - cam.pts_infront : (N, 3)
    nn : dict
        nn[cam.name] must have:
        - pts_all            : (M, 3)
        - group_ids         : (M,)
        - depths_per_kp     : list of per-keypoint depth lists
    """
    fig = plt.figure(figsize=figsize)

    for i, cam in enumerate(cameras):
        ax = fig.add_subplot(
            1, len(cameras), i + 1, projection="3d"
        )

        # All LiDAR points
        ax.scatter(
            cam.pts_infront[:, 0],
            cam.pts_infront[:, 1],
            cam.pts_infront[:, 2],
            s=all_lidar_size,
            c="gray",
            alpha=0.4,
            label="All LiDAR",
        )

        # Neighbor points
        ax.scatter(
            nn[cam.name].pts_all[:, 0],
            nn[cam.name].pts_all[:, 1],
            nn[cam.name].pts_all[:, 2],
            s=neighbor_size,
            c=nn[cam.name].group_ids,
            cmap=cmap,
            alpha=1.0,
            label="Neighbors",
        )

        num_with_neighbors = sum(
            len(d) > 0 for d in nn[cam.name].depths_per_kp
        )

        ax.set_title(
            f"{cam.name} – 3D\n"
            f"{num_with_neighbors} KP w/ neighbors, "
            f"{len(nn[cam.name].pts_all)} pts"
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    return fig
