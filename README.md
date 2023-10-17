# Online Beam Learning With Interference Nulling for Millimeter Wave MIMO Systems
This is the simulation code related to the following article: Y. Zhang, T. Osman, and A. Alkhateeb, "[Online Beam Learning With Interference Nulling for Millimeter Wave MIMO Systems](https://arxiv.org/abs/2209.04509)," in [IEEE Transactions on Wireless Communications]().

# Abstract of the Article
Employing large antenna arrays is a key characteristic of millimeter wave (mmWave) and terahertz communication systems. Due to the hardware constraints and the lack of channel knowledge, codebook based beamforming/combining is normally adopted to achieve the desired array gain. However, most of the existing codebooks focus only on improving the gain of their target user, without taking interference into account. This can incur critical performance degradation in dense networks. In this paper, we propose a sample-efficient online reinforcement learning based beam pattern design algorithm that learns how to shape the beam pattern to null the interfering directions. The proposed approach does not require any explicit channel knowledge or any coordination with the interferers. Simulation results show that the developed solution is capable of learning well-shaped beam patterns that significantly suppress the interference while sacrificing tolerable beamforming/combing gain from the desired user. Furthermore, a hardware proof-of-concept prototype based on mmWave phased arrays is built and used to implement and evaluate the developed online beam learning solutions in realistic scenarios. The learned beam patterns, measured in an anechoic chamber, show the performance gains of the developed framework and highlight a promising machine learning based beam/codebook optimization direction for mmWave and terahertz systems.

# How to reproduce the simulation results?
1. Download all the files of this repository.
2. Run `surrogate_model_dataset_gen.m` in `datasets` directory to generate the power dataset.
3. Run `main.py` in `sm_training_model_based` directory to train the model-based prediction model. For fully-connected neural network-based prediction model, the procedures are exactly the same. (Refer to the detailed instructions in the README file in the subfolder)
4. After step 3 finishes, copy the saved model parameters into `beam_learning_surrogate_model` directory. Then, run `commander_sequential.py` in the same directory.
5. Run `commander_sequential.py` in the `beam_learning_actual_environment` directory to get the results when the learning algorithm is interacting with the actual environment.
6. Run `plot_learning_curves.py`, which will generate figures similar to Fig. 6 in the paper as attached below.

![Figure](https://github.com/YuZhang-GitHub/Interference_Nulling_Beamforming/blob/main/learning_curves.png)

If you have any problems with generating the figure, please contact [Yu Zhang](https://www.linkedin.com/in/yu-zhang-391275181/).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you in any way use this code for research that results in publications, please cite our original article:
> Y. Zhang, T. Osman, and A. Alkhateeb, "[Online Beam Learning With Interference Nulling for Millimeter Wave MIMO Systems](https://arxiv.org/abs/2209.04509)," in IEEE Transactions on Wireless Communications.
