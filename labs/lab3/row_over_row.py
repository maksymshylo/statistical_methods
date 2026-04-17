from typing import Tuple, Dict

import argparse
import json, string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from decimal import Decimal
from tqdm import tqdm

EPS = 10**-320


class StringOverStringDecoder:
    """Overlap strings, apply noise and decode with Gibbs Sampler.

    Note: Each image in alphabet folder has shape (Any, 27).

    Attributes:
        alphabet (list): The list of letters with '|' sign.
        alphabet_dict (dict): The dictionary [letter: image].
        alphabet_imgs (list): The list of alphabet images.
        alphabet_path (str): The path to alphabet folder.
        bigram_probs (np.ndarray): An array of bigram probabilities.
        decoded_string_1 (str): The first decoded string.
        decoded_string_1_im (np.ndarray): The first decoded image.
        decoded_string_2 (str): The second decoded string.
        decoded_string_2_im (np.ndarray): The second decoded image.
        frequencies_dict (dict): The dictionary [letter: frequency].
        frequencies_path (str): The path to frequencies json.
        input_im (np.ndarray): The string over string image.
        n_iter (int): Number of iterations for Gibbs Sampler.
        noise (float): The noise level of bernoulli distribution (0,1).
        noised_im (np.ndarray): The string over string noised image.
        string_1 (str): The first string.
        string_2 (str): The second string.

    Args:
        string_1 (str): The first string.
        string_2 (str): The second string.
        noise (float): The noise level of bernoulli distribution (0,1) add to input_string.
        n_iter (int): Number of iterations for Gibbs Sampler.
        alphabet_path (str): The path to alphabet folder.
        frequencies_path (str): The path to frequencies json.
        seed (int): Seed to debug noised result.
    """

    def __init__(
        self,
        string_1: str,
        string_2: str,
        noise: float,
        n_iter: int,
        alphabet_path: str,
        frequencies_path: str,
        seed: int,
    ):
        """
        Read alphabet folder and frequencies json.
        Convert strings to noised image with overlapped strings, calculate bigram probs.
        """
        self.string_1 = string_1
        self.string_2 = string_2
        self.noise = noise
        self.n_iter = n_iter
        self.alphabet_path = alphabet_path
        self.frequencies_path = frequencies_path

        # set seed for debugging results
        np.random.seed(seed)

        assert 0 <= self.noise <= 1, "Noise level should be in range [0, 1]."

        self.alphabet = list(string.ascii_lowercase + " ")

        # read frequencies json
        with open(self.frequencies_path) as json_file:
            self.frequencies_dict = json.load(json_file)

        self.alphabet_dict = self.__read_alphabet_folder__()
        self.bigram_probs = self.__calculate_bigram_probs__()
        self.add_one_px_space()
        self.alphabet_imgs = list(self.alphabet_dict.values())

        input_string_im_1 = self.string_to_image(string_to_convert=self.string_1)
        input_string_im_2 = self.string_to_image(string_to_convert=self.string_2)

        # make both image the same size
        input_string_im_1, input_string_im_2 = self.make_same_size(
            img1=input_string_im_1, img2=input_string_im_2
        )

        self.input_im = np.logical_or(input_string_im_1, input_string_im_2).astype(int)
        self.noised_im = self.__add_binomial_noise__()

        self.decoded_string_1, self.decoded_string_2 = "", ""
        self.decoded_string_1_im = np.zeros_like(self.noised_im)
        self.decoded_string_2_im = np.zeros_like(self.noised_im)

        print("The input strings has been successfully converted to noised image.")
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
        noise = np.random.binomial(n=1, p=self.noise, size=self.input_im.shape)

        return noise ^ self.input_im

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

    @staticmethod
    def make_same_size(
        img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Make both image the same size."""
        if img1.shape[1] == img2.shape[1]:
            return img1, img2
        elif img1.shape[1] < img2.shape[1]:
            diff_width = img2.shape[1] - img1.shape[1]
            img1 = np.hstack((img1, np.zeros((img1.shape[0], diff_width)))).astype(int)
        else:
            diff_width = img1.shape[1] - img2.shape[1]
            img2 = np.hstack((img2, np.zeros((img2.shape[0], diff_width)))).astype(int)

        return img1, img2

    def add_one_px_space(self):
        """
        Add symbol with one pixel width to generate images with different width.
        """
        self.alphabet += ["|"]
        self.alphabet_dict["|"] = np.zeros(
            (self.alphabet_dict[" "].shape[0], 1), dtype=int
        )
        self.bigram_probs = np.pad(
            self.bigram_probs,
            [(0, 1), (0, 1)],
            mode="constant",
            constant_values=1 / (len(self.alphabet) - 1),
        )

    def __calculate_cond_prob__(
        self,
        letter_1_im: np.ndarray,
        letter_2_im: np.ndarray,
        bigram_letter_prob: float,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate conditional probability of 2 images using bernoulli distribution.

        Args:
            letter_1_im: The first image.
            letter_2_im: The second image.
            mask:
        """
        if self.noise == 0:
            _cond_prob = (letter_1_im ^ letter_2_im) * np.log(self.noise + EPS)
        elif self.noise == 1:
            _cond_prob = (1 ^ letter_1_im ^ letter_2_im) * np.log(EPS)
        else:
            _cond_prob = (letter_1_im ^ letter_2_im) * np.log(self.noise) + (
                1 ^ letter_1_im ^ letter_2_im
            ) * np.log(1 - self.noise)
        # multiplication on mask is needed to sum in cond_probab only
        # that pixels which are black in image at previous iteration
        return np.sum(_cond_prob * mask) + bigram_letter_prob

    def __calculate_prob_sum__(
        self, image: np.ndarray, bigram_p_k: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Calculate sum of prob-s of the image.

        Args:
            image: The input image.
            bigram_p_k: Bigram letter probability.
            mask:

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
                        mask=mask[:, col_idx:cut_till],
                    )
                    # add to inner sum
                    inner_sum[letter_idx] = Decimal(cond_prob).exp() * Decimal(
                        prob_sum[cut_till]
                    )
            prob_sum[col_idx] = np.sum(inner_sum)

        return prob_sum

    def __letter_prob__(
        self,
        image: np.ndarray,
        prob_sum: np.ndarray,
        bigram_p_k: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate probability of the letter.

        Args:
            image: The input image.
            prob_sum: The sum of probability sum.
            bigram_p_k: Bigram letter probability.
            mask:

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
                # multiplication on mask is needed to sum in cond_probab only
                # that pixels which are black in image at previous iteration
                cond_prob = self.__calculate_cond_prob__(
                    letter_1_im=letter_candidate_im,
                    letter_2_im=letter_im,
                    bigram_letter_prob=bigram_p_k[letter_idx],
                    mask=mask[:, :cut_till],
                )
                letter_prob[letter_idx] = Decimal(cond_prob).exp() * Decimal(
                    prob_sum[cut_till]
                )

        return letter_prob

    def generate_string(self, noised_im: np.ndarray, mask: np.ndarray) -> str:
        """Generate string from image."""
        decoded_letters = []
        # assume the first letter is the ' ' symbol and has 0 width
        # the ' ' symbol is the last in alphabet
        width_decoded_letter = 0
        bigram_p_k = self.bigram_probs[-2, :]
        # take noised image piece by piece at each iteration starting from
        # the width of previous decoded letter
        while noised_im[:, width_decoded_letter:].size != 0:
            # cut input_image based on previous letter
            noised_im = noised_im[:, width_decoded_letter:]
            # cut mask based on input_image
            mask = mask[:, width_decoded_letter:]
            # calculate probabilities for the letter
            prob_sum = self.__calculate_prob_sum__(
                image=noised_im, bigram_p_k=bigram_p_k, mask=mask
            )
            letter_probs = self.__letter_prob__(
                image=noised_im, prob_sum=prob_sum, bigram_p_k=bigram_p_k, mask=mask
            )
            # generate letter from probability
            # todo: replace with np.random.Generator.choice
            decoded_letter_idx = np.random.choice(
                a=len(letter_probs), size=1, p=list(letter_probs / letter_probs.sum())
            )[0]
            bigram_p_k = self.bigram_probs[decoded_letter_idx, :]
            # get generated letter
            decoded_letters.append(self.alphabet[decoded_letter_idx])
            # update width
            width_decoded_letter = self.alphabet_imgs[decoded_letter_idx].shape[1]

        return "".join(decoded_letters)

    def decode_noised_image(self) -> None:
        """Decode each string in noised string over another
        one string image with Gibbs Sampler.

        Updates:
            - self.decoded_string_1
            - self.decoded_string_2
            - self.decoded_string_1_im
            - self.decoded_string_2_im.
        """
        # mask is needed to sum pixels which are not black
        # set mask for first iteration as array of ones
        mask = np.ones_like(self.noised_im, dtype=int)
        print("Decoding strings...")
        for iter in range(self.n_iter):
            # generate the first string
            self.decoded_string_1 = self.generate_string(
                noised_im=self.noised_im.copy(), mask=mask
            )
            # convert it to image
            self.decoded_string_1_im = self.string_to_image(
                string_to_convert=self.decoded_string_1
            )
            # update mask
            mask = np.logical_not(self.decoded_string_1_im).astype(int)

            # generate output_string_2
            self.decoded_string_2 = self.generate_string(
                noised_im=self.noised_im.copy(), mask=mask
            )
            # convert it to image
            self.decoded_string_2_im = self.string_to_image(
                string_to_convert=self.decoded_string_2
            )
            # update mask
            mask = np.logical_not(self.decoded_string_2_im).astype(int)
            print(f"Iteration {iter}. String 1: {self.decoded_string_1}; String 2: {self.decoded_string_2}.")


def main():
    # parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--string_1", type=str, required=True, help="The first string to decode."
    )
    parser.add_argument(
        "--string_2", type=str, required=True, help="The second string to decode."
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        required=True,
        help="Noise level of bernoulli distribution.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        required=True,
        help="Number of Gibbs Sampler iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Seed to debug",
    )
    args = parser.parse_args()

    string_over_string_decoder = StringOverStringDecoder(
        string_1=args.string_1,
        string_2=args.string_2,
        noise=args.noise_level,
        n_iter=args.n_iter,
        alphabet_path="lab3/alphabet",
        frequencies_path="lab3/frequencies.json",
        seed=args.seed,
    )

    string_over_string_decoder.decode_noised_image()

    print("Input string 1: ", string_over_string_decoder.string_1)
    print("Input string 2: ", string_over_string_decoder.string_2)

    print("The first decoded string: ", string_over_string_decoder.decoded_string_1)
    print("The second decoded string: ", string_over_string_decoder.decoded_string_2)

    plt.imsave(
        "string_over_string.png", string_over_string_decoder.input_im, cmap="binary"
    )
    plt.imsave(
        "string_over_string_noised.png",
        string_over_string_decoder.noised_im,
        cmap="binary",
    )
    plt.imsave(
        "output_string_1.png",
        string_over_string_decoder.decoded_string_1_im,
        cmap="binary",
    )
    plt.imsave(
        "output_string_2.png",
        string_over_string_decoder.decoded_string_2_im,
        cmap="binary",
    )


if __name__ == "__main__":
    main()
