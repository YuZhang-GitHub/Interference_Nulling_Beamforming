# Model-based Power Prediction
<p align="justify">
  This is the simulation codes and results for the model-based signal and interference prediction.
</p>


# How to run?

We simulate the combinations of the following four cases for the FC-based prediction

- Prediction quantity: Signal or interference.
- Number of antennas: M=8 and M=256.

For clarity, we provide the results for interference prediction with M=256. You can easily modify the codes to run the other three cases.

To generate the results:

1. Run `main.py`, the trained models will be stored in `trained_model` folder.
4. Run `model_eval.py` to get the prediction results.
4. Run `Performance_vs_TrainingSamples.m` to visualize the results (this will be one curve in Fig. 5).

Note:

- Number of realizations can also be changed, which is used for averaging the performance. We perform 100 realizations to generate Fig. 5.
- When the training dataset size changes, it also requires changes in the training schedule to get consistent results. These schedules can be found in the `main.py`.
