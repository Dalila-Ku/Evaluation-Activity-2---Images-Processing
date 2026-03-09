import numpy as np
from scipy.ndimage import convolve, median_filter as scipy_median
from scipy.signal import convolve2d

# Gaussian Filter

GAUSSIAN_KERNEL = np.array(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]], dtype=np.float64
) / 16.0


def gaussian_filter(image: np.ndarray) -> np.ndarray:
    img_f = image.astype(np.float64)
    # scipy.ndimage.convolve is implemented in C via Cython bindings
    result = convolve(img_f, GAUSSIAN_KERNEL, mode='constant', cval=0.0)
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
    gx = convolve(img_f, SOBEL_X, mode='constant', cval=0.0)
    gy = convolve(img_f, SOBEL_Y, mode='constant', cval=0.0)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)


# Median Filter

def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    # scipy.ndimage.median_filter is a compiled C routine
    result = scipy_median(image.astype(np.float64), size=size)
    return np.clip(result, 0, 255).astype(np.uint8)

# Demo — runs only when executing this file directly

if __name__ == '__main__':
    import time
    import numpy as np

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
    print("  filters_numpy_cython.py — Demo")
    print("=" * 50)
    print("\nInput image (8x8):")
    print(test_image)

    # Gaussian
    t0 = time.perf_counter()
    gauss = gaussian_filter(test_image)
    t_gauss = time.perf_counter() - t0
    print(f"\n[Gaussian Filter — SciPy/Cython]  ({t_gauss*1000:.3f} ms)")
    print(gauss)

    # Sobel
    t0 = time.perf_counter()
    sobel = sobel_filter(test_image)
    t_sobel = time.perf_counter() - t0
    print(f"\n[Sobel Filter — SciPy/Cython]  ({t_sobel*1000:.3f} ms)")
    print(sobel)

    # Median
    noisy = test_image.copy()
    noisy[2, 3] = 255
    noisy[4, 4] = 0
    t0 = time.perf_counter()
    med = median_filter(noisy)
    t_median = time.perf_counter() - t0
    print(f"\n[Median Filter — SciPy/Cython]  ({t_median*1000:.3f} ms)")
    print(med)

    print("\nAll NumPy+Cython filters executed successfully.")
