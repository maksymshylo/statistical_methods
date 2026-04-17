import argparse
import json
import string
from decimal import Decimal
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

EPS = 10**-320


class StringImageDecoder:
    """Convert string to image, apply noise and decode it.

    Note: Each image in alphabet folder has shape (Any, 27).

    Attributes:
        input_string (str): A string to convert and decode.
        noise (float): The noise level of bernoulli distribution (0,1).
        alphabet_path (str): The path to alphabet folder.
        frequencies_path (str): The path to frequencies json.
        frequencies_dict (dict): The dictionary [letter: frequency].
        alphabet_dict (dict): The dictionary [letter: image].
        alphabet_imgs (list): The list of alphabet images.

    Args:
        input_string (str): A string to convert and decode.
        noise (float): The noise level of bernoulli distribution (0,1) add to input_string.
        alphabet_path (str): The path to alphabet folder.
        frequencies_path (str): The path to frequencies json.
        seed (int): Seed to debug noised result.
    """

    def __init__(
        self,
        input_string: str,
        noise: float,
        alphabet_path: str,
        frequencies_path: str,
        seed: int,
    ):
        """
        Read alphabet folder and frequencies json.
        Convert input string to image, calculate bigram probs.
        """
        self.input_string = input_string
        self.noise = noise
        self.alphabet_path = alphabet_path
        self.frequencies_path = frequencies_path

        assert 0 <= self.noise <= 1, "Noise level should be in range [0, 1]."

        self.alphabet = list(string.ascii_lowercase + " ")

        # read frequencies json
        with open(self.frequencies_path) as json_file:
            self.frequencies_dict = json.load(json_file)

        self.alphabet_dict = self.__read_alphabet_folder__()
        self.alphabet_imgs = list(self.alphabet_dict.values())

        self.input_string_im = self.string_to_image(string_to_convert=self.input_string)

        # set seed for debugging
        np.random.seed(seed)
        self.noised_im = self.__add_binomial_noise__()
        self.bigram_probs = self.__calculate_bigram_probs__()

        self.decoded_string = ""
        self.decoded_image = np.zeros_like(self.noised_im)
        print("The input string has been successfully converted to image.")
        print("Bigrams have been calculated.")

    def __read_alphabet_folder__(self) -> Dict[str, np.ndarray]:
        """Read images from folder_path.

        Returns:
            A dictionary with letters and its images.
            Each image has shape (Any, 27).
        """
        alphabet_dict = {}
        for letter in self.alphabet[:-1] + ["space"]:
            alphabet_dict[letter] = np.array(
                Image.open(self.alphabet_path + f"/{letter}.png"), dtype=int
            )
        alphabet_dict[" "] = alphabet_dict.pop("space")

        return alphabet_dict

    def string_to_image(self, string_to_convert: str) -> np.ndarray:
        """Convert string to image."""
        string_im = self.alphabet_dict[string_to_convert[0]]
        for letter in string_to_convert[1:]:
            string_im = np.hstack([string_im, self.alphabet_dict[letter]])

        return string_im

    def __add_binomial_noise__(self) -> np.ndarray:
        """Add binomial noise to image."""
        noise = np.random.binomial(n=1, p=self.noise, size=self.input_string_im.shape)

        return noise ^ self.input_string_im

    def __calculate_bigram_probs__(self) -> np.ndarray:
        """Calculate array of bigram (A priori) prob-s. Shape (27, 27). Dtype float64."""
        # create square matrix of frequencies for each bigram
        freq_matrix = np.zeros((len(self.alphabet), len(self.alphabet)), dtype=int)
        for i, letter_i in enumerate(self.alphabet):
            for j, letter_j in enumerate(self.alphabet):
                if letter_i + letter_j in self.frequencies_dict:
                    freq_matrix[i][j] = self.frequencies_dict[letter_i + letter_j]

        # make A priori prob-s from frequencies
        bigram_probs = (freq_matrix.T / freq_matrix.sum(axis=1)).T

        # fill np.log(EPS) where probability == 0
        bigram_probs = np.log(
            bigram_probs,
            out=np.full_like(bigram_probs, np.log(EPS)),
            where=(bigram_probs != 0),
        )

        return bigram_probs

    def __calculate_cond_prob__(
        self,
        letter_1_im: np.ndarray,
        letter_2_im: np.ndarray,
        bigram_letter_prob: float,
    ) -> np.ndarray:
        """
        Calculate conditional probability of 2 images using bernoulli distribution.

        Args:
            letter_1_im: The first image.
            letter_2_im: The second image.
        """
        if self.noise == 0:
            _cond_prob = (letter_1_im ^ letter_2_im) * np.log(self.noise + EPS)
        elif self.noise == 1:
            _cond_prob = (1 ^ letter_1_im ^ letter_2_im) * np.log(EPS)
        else:
            _cond_prob = (letter_1_im ^ letter_2_im) * np.log(self.noise) + (
                1 ^ letter_1_im ^ letter_2_im
            ) * np.log(1 - self.noise)

        return np.sum(_cond_prob) + bigram_letter_prob

    def __calculate_prob_sum__(
        self, image: np.ndarray, bigram_p_k: np.ndarray
    ) -> np.ndarray:
        """Calculate sum of prob-s of the image.

        Args:
            image: The input image.
            bigram_p_k: Bigram letter probability.

        Returns:
            An array of sum of prob-s for each column of image.
            Shape (Any, ). Dtype Decimal.
        """
        image_w = image.shape[1]
        # array of sum of prob-s for each column
        prob_sum = np.full((image_w + 1), fill_value=0, dtype=Decimal)
        # last element always equals 1
        prob_sum[-1] = 1
        # start from the end of image (last column)
        for col_idx in range(image_w - 1, -1, -1):
            # create an array of inner sum for each image
            inner_sum = np.full(
                (len(self.alphabet_imgs)), Decimal(0).normalize(), dtype=Decimal
            )
            # take each image
            for letter_idx, letter_im in enumerate(self.alphabet_imgs):
                # the end of possible letter
                cut_till = col_idx + letter_im.shape[1]
                # calculate conditional probability
                if cut_till <= image_w:
                    letter_candidate_im = image[:, col_idx:cut_till]
                    cond_prob = self.__calculate_cond_prob__(
                        letter_1_im=letter_candidate_im,
                        letter_2_im=letter_im,
                        bigram_letter_prob=bigram_p_k[letter_idx],
                    )
                    # add to inner sum
                    inner_sum[letter_idx] = Decimal(cond_prob).exp() * Decimal(
                        prob_sum[cut_till]
                    )
            prob_sum[col_idx] = np.sum(inner_sum)

        return prob_sum

    def __letter_prob__(
        self, image: np.ndarray, prob_sum: np.ndarray, bigram_p_k: np.ndarray
    ) -> np.ndarray:
        """
        Calculate probability of the letter.

        Args:
            image: The input image.
            prob_sum: The sum of probability sum.
            bigram_p_k: Bigram letter probability.

        Returns:
            An array of sum of prob-s for each column of the image.
            Shape (Any, ). Dtype Decimal.
        """
        image_w = image.shape[1]
        letter_prob = np.zeros(len(self.alphabet_imgs), dtype=Decimal)
        for letter_idx, letter_im in enumerate(self.alphabet_imgs):
            cut_till = letter_im.shape[1]
            if cut_till <= image_w:
                letter_candidate_im = image[:, :cut_till]
                cond_prob = self.__calculate_cond_prob__(
                    letter_1_im=letter_candidate_im,
                    letter_2_im=letter_im,
                    bigram_letter_prob=bigram_p_k[letter_idx],
                )
                letter_prob[letter_idx] = Decimal(cond_prob).exp() * Decimal(
                    prob_sum[cut_till]
                )
        return letter_prob

    def decode_noised_image(self) -> None:
        """Decode noised image.

        Algorithm:
            Assume we have noised_im = "ghft"
            iteration 0:
                1. noised_im = "ghft"
                2. Assume "g" is decoded to "b",
                3. Let's take the width of "b" and cut noised_im
            iteration 1:  noised_im = "hft"
            ...
        """
        decoded_letters = []
        # assume the first letter is the ' ' symbol and has 0 width
        # the ' ' symbol is the last in alphabet
        width_decoded_letter = 0
        bigram_p_k = self.bigram_probs[-1, :]
        # take noised image piece by piece at each iteration starting from
        # the width of previous decoded letter
        noised_im = self.noised_im.copy()
        while noised_im[:, width_decoded_letter:].size != 0:
            # start from the previous letter
            noised_im = noised_im[:, width_decoded_letter:]
            # calculate sum of prob-s assuming
            prob_sum = self.__calculate_prob_sum__(
                image=noised_im, bigram_p_k=bigram_p_k
            )
            letter_probs = self.__letter_prob__(
                image=noised_im, prob_sum=prob_sum, bigram_p_k=bigram_p_k
            )
            # generate letter from probability
            # todo: replace with np.random.Generator.choice
            decoded_letter_idx = np.random.choice(
                a=len(letter_probs), size=1, p=list(letter_probs / letter_probs.sum())
            )[0]
            bigram_p_k = self.bigram_probs[decoded_letter_idx, :]
            # get decoded letter
            decoded_letters.append(self.alphabet[decoded_letter_idx])
            # update width
            width_decoded_letter = self.alphabet_imgs[decoded_letter_idx].shape[1]

        self.decoded_string = "".join(decoded_letters)
        self.decoded_image = self.string_to_image(string_to_convert=self.decoded_string)


def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_string", type=str, required=True, help="Input string to decode."
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        required=True,
        help="Noise level of bernoulli distribution",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="seed to debug",
    )
    args = parser.parse_args()

    string_image_decoder = StringImageDecoder(
        input_string=args.input_string,
        noise=args.noise_level,
        alphabet_path="lab1/alphabet",
        frequencies_path="lab1/frequencies.json",
        seed=args.seed,
    )
    string_image_decoder.decode_noised_image()

    print("Input string: ", string_image_decoder.input_string)
    print("Decoded string: ", string_image_decoder.decoded_string)

    plt.imsave("input_image.png", string_image_decoder.input_string_im, cmap="binary")
    plt.imsave("noised_image.png", string_image_decoder.noised_im, cmap="binary")
    plt.imsave("decoded_image.png", string_image_decoder.decoded_image, cmap="binary")


if __name__ == "__main__":
    main()
