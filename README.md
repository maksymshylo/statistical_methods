# Statistical Methods of Pattern Recognition

**Table of contents**

- [Lab 1. Recognition of a noisy string](#lab-1--recognition-of-a-noisy-image)
- [Lab 2. Recognition of black vertical and horizontal lines](#lab-2--recognition-of-black-vertical-and-horizontal-lines)
- [Lab 3. Gibbs Sampler for recognizing a noisy string over another one](#lab-3--gibbs-sampler-for-recognizing-a-noisy-string-over-another-one)


**Mathematical solutions for the course**

- [solutions.pdf](solutions.pdf)

## Acknowledgements

We sincerely thank the course instructors 
Valerii Krygin ([definability](https://github.com/definability)) and 
Michail I. Schlesinger for their dedication, passion,
and high-quality support throughout the course.

## Setup

To run these applications, you need to have **Python3.12**.

1. Clone repo

2. Create virtual environment.
    ```bash
    python3.12 -m venv .venv
    ```

3. Activate it
    ```bash
    source .venv/bin/activate
    ```

4. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

## Lab 1 – Recognition of a noisy image

### Description
> The program converts a string to a noisy image and then decodes it.

### Usage
```commandline
 $ python3 lab1/decode_string.py --help
usage: decode_string.py [-h] --input_string INPUT_STRING --noise_level NOISE_LEVEL [--seed SEED]

options:
  -h, --help            show this help message and exit
  --input_string INPUT_STRING
                        input string
  --noise_level NOISE_LEVEL
                        noise level of bernoulli distribution
  --seed SEED           seed to debug
```

### Examples

```bash
python3 lab1/decode_string.py --input_string "billy herrington" --noise_level 0.35 --seed 45
```
Decoded string:  "billy herrington"

| Original image                        |              Noisy image               | Decoded image                     |
|---------------------------------------|:--------------------------------------:|-----------------------------------|
| ![](.imgs/lab1/test1/input_image.png) | ![](.imgs/lab1/test1/noised_image.png) | ![](.imgs/lab1/test1/decoded_image.png) |

```bash
python3 lab1/decode_string.py --input_string "van darkholme" --noise_level 0.45 --seed 45
```
Decoded string:  "nde deauff sc"

| Original image                        |              Noisy image               | Decoded image                           |
|---------------------------------------|:--------------------------------------:|-----------------------------------------|
| ![](.imgs/lab1/test2/input_image.png) | ![](.imgs/lab1/test2/noised_image.png) | ![](.imgs/lab1/test2/decoded_image.png) |


## Lab 2 – Recognition of black vertical and horizontal lines

### Description
> The program generates an image with black vertical and horizontal lines,
> apply Bernoulli noise, and denoise it.

##### `Gibbs Sampling`

### Usage
```commandline
$ python3 lab2/gibbs_sampler.py --help
usage: gibbs_sampler.py [-h] [--h H] [--w W] [--n_lines N_LINES]
                        [--noise_level NOISE_LEVEL]
                        [--column_prob COLUMN_PROB] [--n_iter N_ITER]

options:
  -h, --help            show this help message and exit
  --h H                 Height of image.
  --w W                 Width of image.
  --n_lines N_LINES     Number of horizontal and vertical lines.
  --noise_level NOISE_LEVEL
                        Noise level of bernoulli distribution.
  --column_prob COLUMN_PROB
                        Probability of column to be black.
  --n_iter N_ITER       Number of iterations for Gibbs Sampler.
```

#### Examples
```bash
 $ python3 lab2/gibbs_sampler.py --h 250 --w 250 --n_lines 40 --noise_level 0.42 --column_prob 0.5 --n_iter 100
```
> **column accuracy 99.6**
> 
> **row accuracy 99.6**

| Original image                                                    |                            Noisy image                            | Decoded image                                                     |
|-------------------------------------------------------------------|:-----------------------------------------------------------------:|-------------------------------------------------------------------|
| ![](.imgs/lab2/gibbs-sampler-test/input_image_gibbs_sampler.png)  | ![](.imgs/lab2/gibbs-sampler-test/noised_image_gibbs_sampler.png) | ![](.imgs/lab2/gibbs-sampler-test/output_image_gibbs_sampler.png) |


##### `Precise solution`

### Usage
```commandline
 $ python3 lab2/precise_solution.py --help
usage: precise_solution.py [-h] --h H --w W --n_lines N_LINES --noise_level NOISE_LEVEL --column_prob COLUMN_PROB

options:
  -h, --help            show this help message and exit
  --h H                 Height of image.
  --w W                 Width of image.
  --n_lines N_LINES     Number of horizontal and vertical lines.
  --noise_level NOISE_LEVEL
                        Noise level of bernoulli distribution.
  --column_prob COLUMN_PROB
                        Probability of column to be black.
```


#### Examples
```bash
 $ python3 lab2/precise_solution.py --h 250 --w 250 --n_lines 40 --noise_level 0.42 --column_prob 0.5
```
> **column accuracy 96.8**
> 
> **row accuracy 99.2**

| Original image                                                  |                          Noisy image                           | Decoded image                                                  |
|-----------------------------------------------------------------|:--------------------------------------------------------------:|----------------------------------------------------------------|
| ![](.imgs/lab2/precise-solution-test/input_image_precise.png)   | ![](.imgs/lab2/precise-solution-test/noised_image_precise.png) | ![](.imgs/lab2/precise-solution-test/output_image_precise.png) |



## Lab 3 – Gibbs Sampler for recognizing a noisy string over another one

> Note: Lengths of string should be the same.

### Usage
```commandline
 $ python3 lab3/row_over_row.py --help
usage: row_over_row.py [-h] --string_1 STRING_1 --string_2 STRING_2 --noise_level NOISE_LEVEL --n_iter N_ITER [--seed SEED]

options:
  -h, --help            show this help message and exit
  --string_1 STRING_1   The first string to decode.
  --string_2 STRING_2   The second string to decode.
  --noise_level NOISE_LEVEL
                        Noise level of bernoulli distribution.
  --n_iter N_ITER       Number of Gibbs Sampler iterations.
  --seed SEED           Seed to debug

```

#### Examples
```bash
python3 lab3/row_over_row.py --string_1 'deliver' --string_2 'reviled' --noise_level 0.3 --n_iter 10 --seed 67
```
```commandline
Decoding strings...
Iteration 0. String 1: deliver; String 2: reviled.
Iteration 1. String 1: deliver; String 2: reviled.
Iteration 2. String 1: deliver; String 2: reviled.
Iteration 3. String 1: deliver; String 2: reviled.
Iteration 4. String 1: deliver; String 2: reviled.
Iteration 5. String 1: deliver; String 2: reviled.
Iteration 6. String 1: deliver; String 2: reviled.
Iteration 7. String 1: deliver; String 2: reviled.
Iteration 8. String 1: deliver; String 2: reviled.
Iteration 9. String 1: deliver; String 2: reviled.
Input string 1:  deliver
Input string 2:  reviled
The first decoded string:  deliver
The second decoded string:  reviled
```

| Input string over string image               | Noisy string over string image                      | Decoded string 1                          |             Decoded string 2              |
|----------------------------------------------|-----------------------------------------------------|-------------------------------------------|:-----------------------------------------:|
| ![](.imgs/lab3/test1/string_over_string.png) | ![](.imgs/lab3/test1/string_over_string_noised.png) | ![](.imgs/lab3/test1/output_string_1.png) | ![](.imgs/lab3/test1/output_string_2.png) |


```bash
 $ python3 lab3/row_over_row.py --string_1 'hello' --string_2 'world' --noise_level 0.35 --n_iter 25 --seed 67
```

```commandline
Iteration 0. String 1: h|e|||||||||rdo||||; String 2: wo||||||||h||ld.
Iteration 1. String 1: h|e|||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 2. String 1: h|e|||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 3. String 1: h|e|||||||||r||||||||||||o||||; String 2: wo||||||||||||||||||||||ld.
Iteration 4. String 1: h|||z|||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 5. String 1: he||||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 6. String 1: he||||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 7. String 1: h|e|||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 8. String 1: he||||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 9. String 1: h|e|||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 10. String 1: he||||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 11. String 1: h|e|||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 12. String 1: h|e|||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 13. String 1: he||||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 14. String 1: h|e|||||||||r||||||||||||o||||; String 2: wo||||||||h||ld.
Iteration 15. String 1: h|e|||||||||r||||||||||||o||||; String 2: wo||||||||||||||||||||||ld.
Iteration 16. String 1: h||e||||||||r||||||||||||o||||; String 2: world.
Iteration 17. String 1: hello||||; String 2: world.
Iteration 18. String 1: hello||||; String 2: world.
Iteration 19. String 1: hello||||; String 2: world.
Iteration 20. String 1: hello||||; String 2: world.
Iteration 21. String 1: hello||||; String 2: world.
Iteration 22. String 1: hello||||; String 2: world.
Iteration 23. String 1: hello||||; String 2: world.
Iteration 24. String 1: hello||||; String 2: world.
Input string 1:  hello
Input string 2:  world
The first decoded string:  hello||||
The second decoded string:  world
```

| Input string over string image               | Noisy string over string image                      | Decoded string 1                          |             Decoded string 2              |
|----------------------------------------------|-----------------------------------------------------|-------------------------------------------|:-----------------------------------------:|
| ![](.imgs/lab3/test2/string_over_string.png) | ![](.imgs/lab3/test2/string_over_string_noised.png) | ![](.imgs/lab3/test2/output_string_1.png) | ![](.imgs/lab3/test2/output_string_2.png) |


#### Some other examples of possible input strings with the same widths:
```
deliver <=> reviled
animal <=> lamina
depots <=> stoped
diaper <=> repaid
drawer <=> reward
looter <=> retool
murder <=> redrum
redips <=> spider
debut <=> tubed
deeps <=> speed
peels <=> sleep
serif <=> fires
steel <=> leets
````

## Authors

- Maksym Shylo ([@maksymshylo](https://github.com/maksymshylo))
- Ruslan Khomenko ([@Ruslan3584](https://github.com/Ruslan3584))
