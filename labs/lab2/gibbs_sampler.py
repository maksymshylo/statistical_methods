import argparse
import matplotlib.pyplot as plt
from typing import Tuple

import numpy as np

from utilities import *

SEED = 67


def gibbs_sampler(
    image: np.ndarray, noise_level: float, column_prob: np.ndarray, n_iter: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate lines with fixed cols or rows.

    Args:
        image: Input noised image.
        noise_level: Probability of bernoulli distribution.
        column_prob: Probability of  column to be black.
        n_iter: Number of Gibbs Sampler iterations.

    Returns:
        Generated cols and rows.
    """
    # fix rows and cols
    np.random.seed(SEED)
    generated_rows = np.random.randint(2, size=image.shape[0])
    generated_cols = np.zeros_like(generated_rows)
    # calculate apriori probability for set of rows (columns) to be black
    apriori_prob = column_prob ** np.arange(2) * (1 - column_prob) ** (1 - np.arange(2))
    # run Gibbs Sampler
    for _ in range(n_iter):
        # generate cols with fixed rows
        generated_cols = generate_lines(
            image=image,
            fixed_generated=generated_rows,
            noise_level=noise_level,
            apriori_prob=apriori_prob,
            lines_type="cols",
        )
        # generate rows with fixed cols
        generated_rows = generate_lines(
            image=image,
            fixed_generated=generated_cols,
            noise_level=noise_level,
            apriori_prob=apriori_prob,
            lines_type="rows",
        )

    return generated_cols, generated_rows


def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, required=True, help="Height of image.")
    parser.add_argument("--w", type=int, required=True, help="Width of image.")
    parser.add_argument(
        "--n_lines",
        type=int,
        required=True,
        help="Number of horizontal and vertical lines.",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        required=True,
        help="Noise level of bernoulli distribution.",
    )
    parser.add_argument(
        "--column_prob",
        type=float,
        required=True,
        help="Probability of column to be black.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        required=True,
        help="Number of iterations for Gibbs Sampler.",
    )

    args = parser.parse_args()

    image, true_cols, true_rows = generate_image(
        height=args.h,
        width=args.w,
        n_lines=args.n_lines,
    )

    noised_image = apply_bernoulli_noise(image, args.noise_level)

    generated_cols, generated_rows = gibbs_sampler(
        image=noised_image,
        noise_level=args.noise_level,
        column_prob=args.column_prob,
        n_iter=args.n_iter,
    )

    cols_acc, rows_acc = get_accuracy(
        true_cols, true_rows, generated_cols, generated_rows
    )
    print("column accuracy", cols_acc)
    print("row accuracy", rows_acc)

    output_image = build_output_image(generated_cols, generated_rows)

    plt.imsave("input_image_gibbs_sampler.png", image, cmap="binary")
    plt.imsave("noised_image_gibbs_sampler.png", noised_image, cmap="binary")
    plt.imsave("output_image_gibbs_sampler.png", output_image, cmap="binary")


if __name__ == "__main__":
    main()
