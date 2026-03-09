import math

# Helpers

def _clamp(value: float, lo: int = 0, hi: int = 255) -> int:
    """Clamp a value to the [lo, hi] range and return as int."""
    return int(max(lo, min(hi, value)))


def _convolve2d(image: list, kernel: list) -> list:
    height = len(image)
    width = len(image[0])
    k = len(kernel)
    pad = k // 2

    # Initialise output with zeros
    output = [[0] * width for _ in range(height)]

    for y in range(pad, height - pad):
        for x in range(pad, width - pad):
            total = 0.0
            for ky in range(k):
                for kx in range(k):
                    pixel = image[y + ky - pad][x + kx - pad]
                    total += pixel * kernel[ky][kx]
            output[y][x] = _clamp(total)

    return output

# Gaussian Filter

GAUSSIAN_KERNEL = [
    [1 / 16, 2 / 16, 1 / 16],
    [2 / 16, 4 / 16, 2 / 16],
    [1 / 16, 2 / 16, 1 / 16],
]


def gaussian_filter(image: list) -> list:
    return _convolve2d(image, GAUSSIAN_KERNEL)

# Sobel Filter

SOBEL_X = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
]

SOBEL_Y = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
]


def sobel_filter(image: list) -> list:

    gx = _convolve2d(image, SOBEL_X)
    gy = _convolve2d(image, SOBEL_Y)

    height = len(image)
    width = len(image[0])
    output = [[0] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            magnitude = math.sqrt(gx[y][x] ** 2 + gy[y][x] ** 2)
            output[y][x] = _clamp(magnitude)

    return output

# Median Filter

def median_filter(image: list, size: int = 3) -> list:

    height = len(image)
    width = len(image[0])
    pad = size // 2
    output = [[0] * width for _ in range(height)]

    for y in range(pad, height - pad):
        for x in range(pad, width - pad):
            neighbourhood = []
            for ky in range(-pad, pad + 1):
                for kx in range(-pad, pad + 1):
                    neighbourhood.append(image[y + ky][x + kx])
            neighbourhood.sort()
            output[y][x] = neighbourhood[len(neighbourhood) // 2]

    return output

# Demo — runs only when executing this file directly

if __name__ == '__main__':
    import time

    # Build a small 8x8 test image with a known pattern
    test_image = [
        [10,  10,  10,  10,  10,  10,  10,  10],
        [10,  10,  10,  10,  10,  10,  10,  10],
        [10,  10, 200, 200, 200, 200,  10,  10],
        [10,  10, 200,   0,   0, 200,  10,  10],
        [10,  10, 200,   0,   0, 200,  10,  10],
        [10,  10, 200, 200, 200, 200,  10,  10],
        [10,  10,  10,  10,  10,  10,  10,  10],
        [10,  10,  10,  10,  10,  10,  10,  10],
    ]

    print("=" * 50)
    print("  filters_pure_python.py — Demo")
    print("=" * 50)
    print("\nInput image (8x8):")
    for row in test_image:
        print("  " + "  ".join(f"{v:3}" for v in row))

    # Gaussian
    t0 = time.perf_counter()
    gauss = gaussian_filter(test_image)
    t_gauss = time.perf_counter() - t0
    print(f"\n[Gaussian Filter]  ({t_gauss*1000:.3f} ms)")
    for row in gauss:
        print("  " + "  ".join(f"{v:3}" for v in row))

    # Sobel
    t0 = time.perf_counter()
    sobel = sobel_filter(test_image)
    t_sobel = time.perf_counter() - t0
    print(f"\n[Sobel Filter]  ({t_sobel*1000:.3f} ms)")
    for row in sobel:
        print("  " + "  ".join(f"{v:3}" for v in row))

    # Median — add some noise first
    noisy = [row[:] for row in test_image]
    noisy[2][3] = 255   # salt
    noisy[4][4] = 0     # pepper
    t0 = time.perf_counter()
    median = median_filter(noisy)
    t_median = time.perf_counter() - t0
    print(f"\n[Median Filter — noisy input]  ({t_median*1000:.3f} ms)")
    for row in median:
        print("  " + "  ".join(f"{v:3}" for v in row))

    print("\nAll Pure Python filters executed successfully.")
