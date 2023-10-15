# FC-based Power Prediction
<p align="justify">
  This is the simulation codes and results for the fully-connected (FC) neural network based signal and interference prediction.
</p>

# How to run?

We simulate the combinations of the following four cases for the FC-based prediction

- Prediction quantity: Signal or interference.
- Number of antennas: M=8 and M=256.

To generate the results:

1. Modify `options` dictionary in the `main.py` file to adjust to the wanted mode.
2. Modify `train_num_list` to adjust the training dataset size.
3. Run `main.py`.
4. Adjust `check_for_paper.py` accordingly and get the results.

Note:

- Number of realizations can also be changed, which is used for averaging the performance.
- These four values will be the horizontal lines in Fig. 5 of the paper.
