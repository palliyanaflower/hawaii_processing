import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import numpy as np
import os
from matplotlib.widgets import Button

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


# =========================
# Plot helper
# =========================
def plot_all_gps_paths(
    true_cam2_gps_xy,
    true_cam3_gps_xy,
    pred_cam3_gps_xy_aru,
    pred_cam3_gps_xy_pnp,
):
    '''Plot gps position of database items, query items, and retrieved matches'''
    true_cam2_gps_xy = np.asarray(true_cam2_gps_xy)
    true_cam3_gps_xy = np.asarray(true_cam3_gps_xy)
    pred_cam3_gps_xy_aru = np.asarray(pred_cam3_gps_xy_aru)
    pred_cam3_gps_xy_pnp = np.asarray(pred_cam3_gps_xy_pnp)

    # ---- Figure layout ----
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(bottom=0.15)

    ax.scatter(
        true_cam2_gps_xy[:, 0],
        true_cam2_gps_xy[:, 1],
        label="cam2 (true)",
        s=40
    )

    ax.scatter(
        true_cam3_gps_xy[:, 0],
        true_cam3_gps_xy[:, 1],
        label="cam3 (true)",
        s=40,
    )

    ax.scatter(
        pred_cam3_gps_xy_aru[:, 0],
        pred_cam3_gps_xy_aru[:, 1],
        label="cam3 (pred Arun's)",
        s=40,
        marker="*"
    )

    # ax.scatter(
    #     pred_cam3_gps_xy_pnp[:, 0],
    #     pred_cam3_gps_xy_pnp[:, 1],
    #     label="cam3 (pred PnP)",
    #     s=40,
    #     marker="*"
    # )

    ax.set_title("Pose Estimate Evaluation", fontsize=24)
    ax.set_xlabel("X (meters)", fontsize=16)
    ax.set_ylabel("Y (meters)", fontsize=16)
    ax.set_xlim(-400, 500)
    ax.grid(True)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")


def plot_all_retrieval_paths(
    true_cam2_gps_xy_all,
    query_gps_xy,
    retrieved_cam2_gps_xy_k,
):
    '''Plot gps position of database items, query items, and retrieved matches'''
    query_gps_xy = np.asarray(query_gps_xy)
    retrieved_cam2_gps_xy_k = np.asarray(retrieved_cam2_gps_xy_k)

    # ---- Figure layout ----
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.subplots_adjust(bottom=0.15)

    # Query GPS
    ax.scatter(
        query_gps_xy[:, 0],
        query_gps_xy[:, 1],
        label="cam3 (query)",
        s=40
    )

    # True cam2 full path
    ax.scatter(
        true_cam2_gps_xy_all[:, 0],
        true_cam2_gps_xy_all[:, 1],
        label="cam2 true path",
        s=40,
        marker="*"
    )

    # Retrieved cam2 (top-k)
    for q_idx in range(query_gps_xy.shape[0]):
        q = query_gps_xy[q_idx]
        for k_idx in range(retrieved_cam2_gps_xy_k.shape[1]):
            r = retrieved_cam2_gps_xy_k[q_idx, k_idx]

            ax.plot(
                [r[0], q[0]],
                [r[1], q[1]],
                "r--",
                linewidth=1
            )

    ax.set_title("Retrieval Evaluation")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_xlim(-200, 500)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_retrieval_paths(
    true_cam2_gps_xy_all,
    query_gps_xy,
    true_cam2_gps_xy,
    retrieved_cam2_gps_xy_k,
    path_quer,
    path_retr,
    path_true,
    num_matched_list,
    N_init=54
):
    '''Plot gps position of database items, query items, and retrieved matches in interactive GUI'''
    

    query_gps_xy = np.asarray(query_gps_xy)
    true_cam2_gps_xy = np.asarray(true_cam2_gps_xy)
    retrieved_cam2_gps_xy_k = np.asarray(retrieved_cam2_gps_xy_k)

    max_N = len(query_gps_xy) - 1
    N = N_init

    # ---- Figure layout ----
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[2.5, 1],
        height_ratios=[1, 1, 1],
        wspace=0.05,
        hspace=0.1
    )

    ax_xy   = fig.add_subplot(gs[:, 0])
    ax_qimg = fig.add_subplot(gs[0, 1])
    ax_rimg = fig.add_subplot(gs[1, 1])
    ax_timg = fig.add_subplot(gs[2, 1])

    img_axes = [ax_qimg, ax_rimg, ax_timg]

    plt.subplots_adjust(bottom=0.15)

    def load_image_safe(path, shape=(480, 640, 3)):
        if path is None or not os.path.exists(path):
            return np.ones(shape, dtype=np.float32) # Dummy blank image if path not available
        return plt.imread(path)

    def draw():
        ax_xy.clear()

        q = query_gps_xy[N:N+1]
        r = retrieved_cam2_gps_xy_k[N:N+1]

        # Query GPS
        ax_xy.scatter(
            q[:, 0], q[:, 1],
            label="cam3 (query)",
            s=40
        )

        # True cam2 full path
        ax_xy.scatter(
            true_cam2_gps_xy_all[:, 0],
            true_cam2_gps_xy_all[:, 1],
            label="cam2 true path",
            s=40,
            marker="*"
        )

        # Retrieved cam2 (top-k)
        for i in range(r.shape[1]):
            rx = r[:, i, :]

            ax_xy.scatter(
                rx[:, 0], rx[:, 1],
                s=80,
                alpha=0.5,
                # label="cam2 retrieved (top-1)" if i == 0 else None
                label="cam2 retrieved " + str(i)

            )

            for (rx_, ry_), (qx_, qy_) in zip(rx, q):
                if np.isnan(rx_) or np.isnan(ry_):
                    continue
                ax_xy.plot(
                    [rx_, qx_],
                    [ry_, qy_],
                    "r--",
                    linewidth=1
                )

        ax_xy.set_title(f"Retrieval Evaluation — N = {N}, Num Matches: {num_matched_list[N]}")
        ax_xy.set_xlabel("X (meters)")
        ax_xy.set_ylabel("Y (meters)")
        ax_xy.set_xlim(-500, 500)
        ax_xy.grid(True)
        ax_xy.legend()
        ax_xy.set_aspect("equal", adjustable="box")


        # ---- Images ----
        img_shape = plt.imread(path_quer[N]).shape
        images = [
            load_image_safe(path_quer[N]),
            load_image_safe(path_retr[N][0], img_shape),
            load_image_safe(path_true[N]),
        ]

        titles = [
            "Query (cam3)",
            "Retrieved (cam2)",
            "True (cam2)"
        ]

        for ax, img, title in zip(img_axes, images, titles):
            ax.clear()
            if img is not None:
                ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        fig.canvas.draw_idle()

    # ---- Keyboard controls ----
    def on_key(event):
        nonlocal N
        if event.key == "left":
            N = max(0, N - 1)
            draw()
        elif event.key == "right":
            N = min(max_N, N + 1)
            draw()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ---- Buttons ----
    ax_prev = plt.axes([0.30, 0.05, 0.15, 0.06])
    ax_next = plt.axes([0.55, 0.05, 0.15, 0.06])

    btn_prev = Button(ax_prev, "Prev")
    btn_next = Button(ax_next, "Next")

    def prev(event):
        nonlocal N
        N = max(0, N - 1)

        # Print paths
        print("\nQuery")
        print(path_quer[N])
        print("\nRetrieved")
        for p in path_retr[N]:
            print(p)

        draw()

    def next_(event):
        nonlocal N
        N = min(max_N, N + 1)

        # Print paths
        print("\nQuery")
        print(path_quer[N])
        print("\nRetrieved")
        for p in path_retr[N]:
            print(p)

        draw()

    btn_prev.on_clicked(prev)
    btn_next.on_clicked(next_)

    draw()
    plt.show()
