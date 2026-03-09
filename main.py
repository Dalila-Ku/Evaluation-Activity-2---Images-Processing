import sys
import time
import math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import filters_pure_python as fp
import filters_numpy as fn
import filters_numpy_cython as fc

# Helpers

def load_image(path: str | None) -> np.ndarray:
    """Load a grayscale image from disk or generate a synthetic test image."""
    if path:
        img = Image.open(path).convert('L')
        return np.array(img, dtype=np.uint8)

    # ---- Synthetic 256x256 test image ----
    print("No image path provided — generating a synthetic test image.")
    size = 256
    img = Image.new('L', (size, size), color=40)
    draw = ImageDraw.Draw(img)

    # Geometric shapes to give the filters interesting structure
    draw.ellipse([60, 60, 196, 196], fill=200)
    draw.rectangle([90, 90, 165, 165], fill=80)
    draw.line([0, 0, size, size], fill=255, width=3)
    draw.line([size, 0, 0, size], fill=255, width=3)

    arr = np.array(img, dtype=np.uint8)

    # Add salt-and-pepper noise so the median filter has something to do
    rng = np.random.default_rng(42)
    noise_mask = rng.random(arr.shape)
    arr[noise_mask < 0.04] = 0    # pepper
    arr[noise_mask > 0.96] = 255  # salt
    return arr


def numpy_to_list(arr: np.ndarray) -> list:
    """Convert a 2-D numpy uint8 array to a list-of-lists for pure-Python filters."""
    return arr.tolist()


def list_to_numpy(lst: list) -> np.ndarray:
    """Convert a list-of-lists back to a numpy uint8 array."""
    return np.array(lst, dtype=np.uint8)


def time_it(func, *args, repeats: int = 1) -> tuple:
    """
    Run *func* with *args* for *repeats* iterations and return
    (result_of_first_call, average_elapsed_seconds).
    """
    result = None
    start = time.perf_counter()
    for _ in range(repeats):
        result = func(*args)
    elapsed = (time.perf_counter() - start) / repeats
    return result, elapsed

# Benchmarking

def run_benchmarks(image_np: np.ndarray):
    """
    Run all filters with all three implementations and collect timing data.

    Returns
    -------
    results : dict  {filter_name: {impl_name: (output_np, time_s)}}
    """
    image_list = numpy_to_list(image_np)
    results = {}

    filters = {
        'Gaussian': {
            'Pure Python': (fp.gaussian_filter, image_list, 'list'),
            'NumPy':       (fn.gaussian_filter, image_np,   'numpy'),
            'NumPy+Cython':(fc.gaussian_filter, image_np,   'numpy'),
        },
        'Sobel': {
            'Pure Python': (fp.sobel_filter, image_list, 'list'),
            'NumPy':       (fn.sobel_filter, image_np,   'numpy'),
            'NumPy+Cython':(fc.sobel_filter, image_np,   'numpy'),
        },
        'Median': {
            'Pure Python': (fp.median_filter, image_list, 'list'),
            'NumPy':       (fn.median_filter, image_np,   'numpy'),
            'NumPy+Cython':(fc.median_filter, image_np,   'numpy'),
        },
    }

    for filter_name, impls in filters.items():
        results[filter_name] = {}
        for impl_name, (func, data, dtype) in impls.items():
            print(f"  Running {filter_name} [{impl_name}] ...", end=' ', flush=True)
            output, elapsed = time_it(func, data)
            if dtype == 'list':
                output = list_to_numpy(output)
            results[filter_name][impl_name] = (output, elapsed)
            print(f"{elapsed:.4f}s")

    return results

# Visualisation

def save_visualisation(image_np: np.ndarray, results: dict, output_path: str = 'results.png'):
    """
    Create a grid figure:
      Rows    → Original + 3 filters
      Columns → Pure Python | NumPy | NumPy+Cython
    """
    impls = ['Pure Python', 'NumPy', 'NumPy+Cython']
    filter_names = ['Gaussian', 'Sobel', 'Median']

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#1a1a2e')

    # Title
    fig.suptitle('Image Processing Unit 2 — Filter Comparison',
                 fontsize=18, fontweight='bold', color='white', y=0.98)

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.15)

    # Row 0 — Original image (span all columns)
    ax_orig = fig.add_subplot(gs[0, :])
    ax_orig.imshow(image_np, cmap='gray', vmin=0, vmax=255)
    ax_orig.set_title('Original Grayscale Image', color='white', fontsize=13, pad=8)
    ax_orig.axis('off')

    colours = {'Pure Python': '#e94560', 'NumPy': '#0f3460', 'NumPy+Cython': '#533483'}
    labels = {'Pure Python': '#e94560', 'NumPy': '#16213e', 'NumPy+Cython': '#533483'}

    for row_idx, fname in enumerate(filter_names, start=1):
        for col_idx, impl in enumerate(impls):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            img_out, t = results[fname][impl]
            ax.imshow(img_out, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'{fname} — {impl}\n{t*1000:.1f} ms',
                         color='white', fontsize=9, pad=4)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(list(colours.values())[col_idx])
                spine.set_linewidth(2)

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nVisualisation saved → {output_path}")

# Performance table

def print_table(results: dict):
    """Print a formatted performance comparison table to stdout."""
    impls = ['Pure Python', 'NumPy', 'NumPy+Cython']
    col_w = 18

    header = f"{'Filter':<14}" + "".join(f"{i:>{col_w}}" for i in impls)
    sep = '-' * len(header)

    print("\n" + "=" * len(header))
    print("  PERFORMANCE COMPARISON  (average wall-clock time)")
    print("=" * len(header))
    print(header)
    print(sep)

    for fname, impls_data in results.items():
        times = {i: impls_data[i][1] for i in impls}
        row = f"{fname:<14}"
        for i in impls:
            t = times[i]
            row += f"{t*1000:>{col_w-3}.2f} ms   "
        print(row)

        # Speedup relative to Pure Python
        base = times['Pure Python']
        speedup_row = f"{'  speedup':<14}"
        for i in impls:
            sp = base / times[i]
            speedup_row += f"{'x'+f'{sp:.1f}':>{col_w}}"
        print(speedup_row)
        print(sep)

    print()

# Main

if __name__ == '__main__':
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    image_np = load_image(img_path)

    print(f"\nImage shape: {image_np.shape}  dtype: {image_np.dtype}")
    print("\nRunning benchmarks …\n")

    results = run_benchmarks(image_np)
    print_table(results)
    save_visualisation(image_np, results, output_path='results.png')
