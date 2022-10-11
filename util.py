import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
import json

# mpl.use('Agg')
import torch


def viz_masks(out_name, imname='masks.jpg'):
    """Visualize the masks.
    """
    with open(out_name, 'r') as reader:
        data = json.load(reader)

    fig = plt.figure(figsize=(16 * 4, 1 * 4))
    gs = mpl.gridspec.GridSpec(1, 16)

    for maski, mask in enumerate(data):
        plt.subplot(gs[0, maski])
        plt.imshow(np.array(mask).T, origin='lower', vmin=0, vmax=1)
        plt.title(f'N Occupied cells: {np.array(mask).sum()}')

    plt.tight_layout()
    print('saving', imname)
    plt.savefig(imname)
    plt.close(fig)


def get_grid(point_cloud_range, voxel_size):
    lower = np.array(point_cloud_range[:(len(point_cloud_range) // 2)])
    upper = np.array(point_cloud_range[(len(point_cloud_range) // 2):])

    dx = np.array(voxel_size)
    bx = lower + dx / 2.0
    nx = ((upper - lower) / dx).astype(int)

    return dx, bx, nx


def get_rot(h):
    return np.array([
        [np.cos(h), -np.sin(h)],
        [np.sin(h), np.cos(h)],
    ])


def get_area(points, x, y, bounds=60):
    pts = points
    keptX = np.logical_and(pts[:, 0] <= x + bounds, x - bounds <= pts[:, 0])
    keptY = np.logical_and(pts[:, 2] <= y + bounds, y - bounds <= pts[:, 2])
    keptZ = pts[:, 1] <= 0.001
    kept = np.logical_and(keptX, keptY)
    kept = np.logical_and(kept, keptZ)
    pts = pts[kept]
    return pts[:, (0, 2)]


def get_corners(box, lw):
    l, w = lw
    simple_box = np.array([
        [-l / 2., -w / 2.],
        [l / 2., -w / 2.],
        [l / 2., w / 2.],
        [-l / 2., w / 2.],
    ])
    h = np.arctan2(box[3], box[2])
    #     print(h)
    rot = get_rot(h)
    simple_box = np.dot(simple_box, rot)
    simple_box += box[:2]
    return simple_box


def objects2frame(history, center, toworld=False):
    """A sphagetti function that converts from global
    coordinates to "center" coordinates or the inverse.
    It has no for loops but works on batchs.
    """
    N, A, B = history.shape
    theta = np.arctan2(center[3], center[2])
    if not toworld:
        newloc = history[:, :, :2] - center[:2].reshape((1, 1, 2))
        rot = get_rot(-theta + np.pi / 2)
        #         newh = -np.arctan2(history[:, :, 3], history[:, :, 2]) + theta
        newh = np.arctan2(history[:, :, 3], history[:, :, 2]) - theta
        #         newh = theta
        newloc = np.dot(newloc.reshape((N * A, 2)), rot).reshape((N, A, 2))
    else:
        rot = get_rot(theta)
        newh = np.arctan2(history[:, :, 3], history[:, :, 2]) + theta
        newloc = np.dot(history[:, :, :2].reshape((N * A, 2)),
                        rot).reshape((N, A, 2))
    newh = np.stack((np.cos(newh), np.sin(newh)), 2)
    if toworld:
        newloc += center[:2]
    return np.append(newloc, newh, axis=2)


#     print(history)


def make_rgba(probs, color):
    H, W = probs.shape
    return np.stack((
        np.full((H, W), color[0]),
        np.full((H, W), color[1]),
        np.full((H, W), color[2]),
        probs,
    ), 2)


def render_observation(x):
    # road
    if torch.is_tensor(x):
        x = x.numpy()
    showimg = make_rgba(x[0].T, (1.00, 0.50, 0.31))
    plt.imshow(showimg, origin='lower')

    # road div
    showimg = make_rgba(x[1].T, (159. / 255., 0.0, 1.0))
    plt.imshow(showimg, origin='lower')

    # lane div
    showimg = make_rgba(x[2].T, (0.0, 0.0, 1.0))
    plt.imshow(showimg, origin='lower')

    # objects
    showimg = make_rgba(x[3].T, (0.0, 0.0, 0.0))
    plt.imshow(showimg, origin='lower')

    # ego
    showimg = make_rgba(x[4].T, (0.0, 0.5, 0.0))
    plt.imshow(showimg, origin='lower')
    plt.grid(b=None)
    plt.xticks([])
    plt.yticks([])


def render_observations_and_traj(x, y, gts):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    # road
    showimg = make_rgba(x[0].T, (1.00, 0.50, 0.31))
    plt.imshow(showimg, origin='lower')

    # road div

    showimg = make_rgba(x[1].T, (159. / 255., 0.0, 1.0))
    plt.imshow(showimg, origin='lower')

    # lane div
    showimg = make_rgba(x[2].T, (0.0, 0.0, 1.0))
    plt.imshow(showimg, origin='lower')

    # objects
    showimg = make_rgba(x[3].T, (0.0, 0.0, 0.0))
    plt.imshow(showimg, origin='lower')

    # ego
    showimg = make_rgba(x[4].T, (0.0, 0.5, 0.0))
    plt.imshow(showimg, origin='lower')

    # showimg = make_rgba(y.sum(axis=0).numpy().T, (0.5, 0.5, 0.0))
    # plt.imshow(showimg, origin='lower')

    if gts is not None:
        for i, gt in enumerate(gts):
            temp = np.zeros(x[4].T.shape)
            for ii, (pt1, pt2) in enumerate(zip(gt[:-1], gt[1:])):
                if np.sum(pt1) == 0 or np.sum(pt2) == 0:
                    continue
                if ii == len(gt[:-1]) - 1:
                    temp = cv2.arrowedLine(temp, pt1[::-1], pt2[::-1], (1.0, 0, 0), thickness=2, tipLength=1)
                else:
                    temp = cv2.line(temp, pt1[::-1], pt2[::-1], (1.0, 0, 0), thickness=2)
                # temp = cv2.circle(temp, (pt1[1], pt1[0]), 3, (1.0, 0, 0), thickness=1)
                # temp = cv2.circle(temp, (pt2[1], pt2[0]), 3, (1.0, 0, 0), thickness=1)

            showimg = make_rgba(temp, (0.0, 0.5, 0.5))
            plt.imshow(showimg, origin="lower", alpha=1 - np.tanh(i * 1 / len(gts)))

    if y is not None:
        temp = np.zeros(x[4].T.shape)
        pts = y.detach().cpu().numpy().astype(int)
        for pt in pts:
            temp = cv2.circle(temp, (pt[1], pt[0]), 2, (1.0, 0, 0), thickness=-1)
        showimg = make_rgba(temp, (0.5, 0.5, 0.0))
        plt.imshow(showimg, origin='lower', alpha=1)
    plt.grid(b=None)
    plt.xticks([])
    plt.yticks([])


def render_layers(x, y):
    def style_ax():
        plt.grid(b=None)
        plt.xticks([])
        plt.yticks([])

    fig = plt.figure(figsize=(6 * (x.shape[0] + 2), 6))
    gs = mpl.gridspec.GridSpec(1, x.shape[0] + 2)

    for i in range(x.shape[0]):
        plt.subplot(gs[0, i])
        plt.imshow(x[i].T, origin='lower', vmin=0, vmax=1)
        style_ax()

    plt.subplot(gs[0, i + 1])
    traj_img = np.zeros((256, 256))
    for point in y:
        point = point.detach().numpy().astype(int)
        traj_img[point[0], point[1]] = 1.0
    plt.imshow(traj_img, origin='lower', vmin=0, vmax=1)
    style_ax()

    plt.subplot(gs[0, i + 2])
    render_observation(x)

    plt.tight_layout()
