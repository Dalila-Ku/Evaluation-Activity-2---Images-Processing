
import numpy as np

# Gaussian Filter
GAUSSIAN_KERNEL = np.array(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]], dtype=np.float64
) / 16.0

def _convolve2d_numpy(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.shape[0]
    pad = k // 2
    h, w = image.shape

    # Pad image with zeros on all sides
    padded = np.pad(image, pad, mode='constant', constant_values=0)

    # Use stride tricks to create a view of all (k x k) patches
    shape = (h, w, k, k)
    strides = padded.strides + padded.strides
    patches = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # Dot each patch with the flipped kernel (correlation = convolution for symmetric kernels)
    output = np.einsum('hwij,ij->hw', patches, kernel)
    return output


def gaussian_filter(image: np.ndarray) -> np.ndarray:
    img_f = image.astype(np.float64)
    result = _convolve2d_numpy(img_f, GAUSSIAN_KERNEL)
    return np.clip(result, 0, 255).astype(np.uint8)

# Sobel Filter

SOBEL_X = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], dtype=np.float64
)

SOBEL_Y = np.array(
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]], dtype=np.float64
)

def sobel_filter(image: np.ndarray) -> np.ndarray:
    img_f = image.astype(np.float64)
    gx = _convolve2d_numpy(img_f, SOBEL_X)
    gy = _convolve2d_numpy(img_f, SOBEL_Y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)

# Median Filter

def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    pad = size // 2
    h, w = image.shape
    padded = np.pad(image, pad, mode='constant', constant_values=0)

    # Build (H, W, size, size) patch array using stride tricks
    shape = (h, w, size, size)
    strides = padded.strides + padded.strides
    patches = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # Reshape to (H, W, size*size) then take median along last axis
    flat_patches = patches.reshape(h, w, -1)
    result = np.median(flat_patches, axis=2)
    return result.astype(np.uint8)

# Demo — runs only when executing this file directly

if __name__ == '__main__':
    import time
    import numpy as np

    # Same 8x8 test image as the pure-python demo
    test_image = np.array([
        [10,  10,  10,  10,  10,  10,  10,  10],
        [10,  10,  10,  10,  10,  10,  10,  10],
        [10,  10, 200, 200, 200, 200,  10,  10],
        [10,  10, 200,   0,   0, 200,  10,  10],
        [10,  10, 200,   0,   0, 200,  10,  10],
        [10,  10, 200, 200, 200, 200,  10,  10],
        [10,  10,  10,  10,  10,  10,  10,  10],
        [10,  10,  10,  10,  10,  10,  10,  10],
    ], dtype=np.uint8)

    print("=" * 50)
    print("  filters_numpy.py — Demo")
    print("=" * 50)
    print("\nInput image (8x8):")
    print(test_image)

    # Gaussian
    t0 = time.perf_counter()
    gauss = gaussian_filter(test_image)
    t_gauss = time.perf_counter() - t0
    print(f"\n[Gaussian Filter]  ({t_gauss*1000:.3f} ms)")
    print(gauss)

    # Sobel
    t0 = time.perf_counter()
    sobel = sobel_filter(test_image)
    t_sobel = time.perf_counter() - t0
    print(f"\n[Sobel Filter]  ({t_sobel*1000:.3f} ms)")
    print(sobel)

    # Median — add salt-and-pepper noise
    noisy = test_image.copy()
    noisy[2, 3] = 255
    noisy[4, 4] = 0
    t0 = time.perf_counter()
    med = median_filter(noisy)
    t_median = time.perf_counter() - t0
    print(f"\n[Median Filter — noisy input]  ({t_median*1000:.3f} ms)")
    print(med)

    print("\nAll NumPy filters executed successfully.")
