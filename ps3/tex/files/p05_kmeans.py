from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

def k_means(points, K):
    """
    Arguments:
        - points: (M, N)
    """
    eps = 1e-3
    M, N = points.shape
    belongsto = np.zeros(M)
    # randomly select K points as centroids
    indices = np.arange(M)
    indices = np.random.choice(indices, size=K)
    # (K, N)
    centroids = points[indices].copy()

    max_iter = 300
    it = 0
    prev_dist, dist = None, None
    while it < max_iter and (prev_dist is None or abs(dist - prev_dist) > 1e-3):
        it += 1
        prev_dist = dist
        # assign points to centroids
        # compute distance
        # (M, K, N)
        diff = points[:, None] - centroids[None]
        dist = np.linalg.norm(diff, axis=2)
        belongsto = np.argmin(dist, axis=1)


        # update centroids
        for i in range(K):
            group = points[belongsto == i]
            centroids[i] = group.mean(axis=0)

        dist = np.linalg.norm(points - centroids[belongsto], axis=1).sum()
        print("Iter: {}, Dist: {}".format(it, dist))

    return centroids

if __name__ == '__main__':
    small_path = '../data/peppers-small.tiff'
    large_path = '../data/peppers-large.tiff'

    small = imread(small_path)
    large = imread(large_path)

    H, W, C = small.shape
    K = 16
    points = small.reshape(H * W, C)
    centroids = k_means(points.astype(float), K)

    diff = large[:, :, None] - centroids[None, None]
    dist= np.linalg.norm(diff, axis=3)
    # (H, W)
    indices = np.argmin(dist, axis=2)
    large_new = centroids[indices].astype(np.uint8)
    plt.subplot(1, 2, 1)
    plt.imshow(large)
    plt.subplot(1, 2, 2)
    plt.imshow(large_new)
    plt.show()

