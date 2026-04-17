import numpy as np
from numba import njit

from typing import Tuple

SEED = 67


def generate_image(
    height: int, width: int, n_lines: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate image with black lines.

    Note: The number of vertical and horizontal lines
          that should be black in the input image is randomly selected.

    Returns:
        Generated image, black columns and rows.
    """
    image = np.zeros((height, width), dtype=int)

    true_rows = np.zeros((height,), dtype=int)
    true_cols = np.zeros((width,), dtype=int)

    np.random.seed(SEED)
    rows_indices = np.random.choice(range(height), n_lines, replace=False)
    cols_indices = np.random.choice(range(width), n_lines, replace=False)

    true_rows[rows_indices] = 1
    true_cols[cols_indices] = 1

    image[:, cols_indices] = 1
    image[rows_indices, :] = 1

    return image, true_cols, true_rows


def apply_bernoulli_noise(image: np.ndarray, noise_level: float) -> np.ndarray:
    np.random.seed(SEED)
    ksi = np.random.binomial(size=image.size, n=1, p=noise_level).reshape(image.shape)
    return ksi ^ image


def build_output_image(generated_cols: np.ndarray, generated_rows: np.ndarray):
    """Build image from generated_cols and generated_rows"""
    height, width = len(generated_rows), len(generated_cols)

    output_img = np.zeros((height, width), dtype=int)

    out_cols_ind = np.arange(0, width)[generated_cols != 0]
    out_rows_ind = np.arange(0, height)[generated_rows != 0]

    output_img[:, out_cols_ind] = 1
    output_img[out_rows_ind, :] = 1

    return output_img


def get_accuracy(
    true_cols: np.ndarray,
    true_rows: np.ndarray,
    output_cols: np.ndarray,
    output_rows: np.ndarray,
) -> Tuple[float, float]:
    """Calculate percentage of matched columns and rows."""
    cols_acc = ((output_cols == true_cols).sum() / len(true_cols)) * 100
    rows_acc = ((output_rows == true_rows).sum() / len(true_rows)) * 100
    return cols_acc, rows_acc


@njit
def random_choice(distribution):
    # generate random number from a given distribution
    np.random.seed(SEED)
    r = np.random.uniform(0, 1)
    s = 0
    for item, prob in enumerate(distribution):
        s += prob
        if s >= r:
            return item

    return item


@njit
def generate_lines(
    image: np.ndarray,
    fixed_generated: np.ndarray,
    noise_level: float,
    apriori_prob: np.ndarray,
    lines_type: str,
):
    """Generate rows/cols based on fixed_generated.

    Args:
        image: Input noised image.
        fixed_generated: Previously generated black rows/columns.
        noise_level: Probability of bernoulli distribution.
        apriori_prob: Apriori probability for set of rows/columns to be black.
        lines_type: Type of lines to generate ("cols" or "rows").

    Returns:
        An array of the most possible positions of black lines.
    """
    height, width = image.shape

    labels = np.arange(2)
    # opposite shapes for cols and rows cases
    if lines_type == "cols":
        dim1, dim2 = width, height
        _image = image
    elif lines_type == "rows":
        dim1, dim2 = height, width
        _image = image.T
    else:
        raise ValueError("Wrong line_type!")

    generated_lines = np.zeros(dim1, dtype=np.int64)

    for j in range(dim1):
        # conditional probability for image with fixed rows and columns
        line_cond_prob = np.zeros((2, dim2), dtype=np.float64)
        for i in range(dim2):
            label = np.logical_or(labels, fixed_generated[i])
            line_cond_prob[:, i] = (_image[i, j] ^ label) * noise_level + (
                _image[i, j] ^ (1 - label)
            ) * (1 - noise_level)
        # probability of row/column to be black (with opposite fixed row/column)
        line_prob = (
            np.array(
                [
                    np.prod(line_cond_prob[0, :]),
                    np.prod(line_cond_prob[1, :]),
                ]
            )
            * apriori_prob
        )
        # fill with 1 the index of probable row/column
        generated_lines[j] = random_choice(line_prob / np.sum(line_prob))

    return generated_lines
