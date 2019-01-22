This notebook provides a simple example of how to setup an MDP and use Value Iteration to find the optimal policy and the expected values.

## Getting Started 

- pymdptoolbox only supports python 2.7, 3.2, 3.3, 3.4
- Install dependencies jupyter, pymdptoolbox, and numpy

```
pip install -r requirements.txt
```

- Run jupyter notebook for practice

```
jupyter-notebook pymdptoolbox_example.ipynb
```

## How to complete the homework

open pymdp_DieN.py, change inital setting according to the question.
For example
Input: N = 6, isBadSide = {1,1,1,0,0,0}, Output: 2.5833
```
# Calculate States
N = 6
isBadSide = np.array([1,1,1,0,0,0])
```
Then run below command in the terminal
```
python pymdp_DieN.py
```
The largest values is the answer.

## Contribute

If you have suggestions for improvements or bug fixes, feel free to submit a [pull request](https://help.github.com/articles/creating-a-pull-request/) or create an [issue](https://github.com/rldm/rldm_tutorials/issues).
