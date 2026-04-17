import argparse
import numpy as np
import matplotlib.pyplot as plt

from utilities import *

SEED = 67


def generate_precise_columns(
    image: np.ndarray, noise_level: float, column_prob: np.ndarray
) -> np.ndarray:
    """
    Note: Works fast for relatively small images.

    Args:
        image
        noise_level: Probability of bernoulli distribution.
        column_prob

    Returns:
        Precise positions of black columns.
    """
    np.random.seed(SEED)
    height, width = image.shape
    # labels of column 0 or 1
    labels = np.arange(2)
    # column probability
    cols_probs = np.zeros((2, width))
    # temporary array of probabilities to get image with given columns and rows
    cond_prob_arr = np.zeros((2, height), dtype=np.float64)
    cond_prob = np.zeros((2))
    for c in labels:
        for j in range(width):
            for r_j in labels:
                for i in range(height):
                    cond_prob_arr[r_j, i] = (image[i, j] ^ (c or r_j)) * noise_level + (
                        image[i, j] ^ (1 - c or r_j)
                    ) * (1 - noise_level)
                cond_prob[r_j] = np.prod(cond_prob_arr[r_j, :]) * column_prob[r_j]
            # sum of all possible values of r_j
            cols_probs[c, j] = np.sum(cond_prob)

    # precise position of black columns
    generated_cols = np.zeros((width), dtype=int)
    # get columns by generating from distribution
    for j in range(width):
        generated_cols[j] = np.random.choice(
            labels, p=cols_probs[:, j] / np.sum(cols_probs[:, j])
        )

    return generated_cols


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
        required=True,
        type=float,
        help="Noise level of bernoulli distribution.",
    )
    parser.add_argument(
        "--column_prob",
        type=float,
        required=True,
        help="Probability of column to be black.",
    )

    args = parser.parse_args()

    image, true_cols, true_rows = generate_image(
        height=args.h, width=args.w, n_lines=args.n_lines
    )
    noised_image = apply_bernoulli_noise(image, args.noise_level)

    pc_r = args.column_prob ** np.arange(2) * (1 - args.column_prob) ** (
        1 - np.arange(2)
    )
    generated_cols = generate_precise_columns(
        image=noised_image, noise_level=args.noise_level, column_prob=pc_r
    )
    # generating rows by fixing columns
    generated_rows = generate_lines(
        image=noised_image,
        fixed_generated=generated_cols,
        noise_level=args.noise_level,
        apriori_prob=pc_r,
        lines_type="rows",
    )

    cols_acc, rows_acc = get_accuracy(
        true_cols, true_rows, generated_cols, generated_rows
    )
    print("column accuracy", cols_acc)
    print("row accuracy", rows_acc)

    output_image = build_output_image(generated_cols, generated_rows)

    plt.imsave("input_image_precise.png", image, cmap="binary")
    plt.imsave("noised_image_precise.png", noised_image, cmap="binary")
    plt.imsave("output_image_precise.png", output_image, cmap="binary")


if __name__ == "__main__":
    main()
