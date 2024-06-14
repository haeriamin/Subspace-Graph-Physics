# Subspace Graph Physics: <br /> Accurate and Real-Time Physics Simulation Approach
#### [[Paper](https://www.sciencedirect.com/science/article/pii/S0952197624009230)]


Subspace Graph Physics is an <strong>accurate, real-time</strong> engineering approach for large-scale 3D/2D physics simulations.
This is a computationally efficient version of "Learning To Simulate" developed by [DeepMind](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate).
This approach can work with a single desktop GPU (with ~10GB vRAM) in training and can perform 2-3 orders of magnitude faster than physics-based methods (with similar accuracy) in inference on CPU.

<img src="https://github.com/haeriamin/files/blob/master/excav_ml_4.gif" alt="drawing" width="820">


# Highlights

* We train the graph network (GN) model in subspace by performing Principal Component Analysis (PCA).

* PCA enables GN to be trained using a single desktop GPU with moderate VRAM for large 3D configurations.

* The training datasets can be generated by our efficient and accurate [Material Point Method](https://github.com/haeriamin/MPM-NGF) (MPM).

* The rollout runtime is under 1 sec/sec, and the training runtime is 60 global-step/sec (on NVIDIA RTX 3080).

* The particle positions and velocities, and rigid body interaction forces are compared in the video above.


# Install and Run Demo

* Install

    * Install Python (tested on version 3.7)

    * (Optional) [Install TensorFlow 1.15 for NVIDIA RTX30 GPUs (without docker or CUDA install)](https://www.pugetsystems.com/labs/hpc/How-To-Install-TensorFlow-1-15-for-NVIDIA-RTX30-GPUs-without-docker-or-CUDA-install-2005/)

    * Run

        ```bash
        pip install -r requirements.txt
        ```

* Train

    ```python
    python -m learning_to_simulate.train --mode=train --eval_split=train --batch_size=2 --data_path=./learning_to_simulate/datasets/Excavation_PCA --model_path=./learning_to_simulate/models/Excavation_PCA
    ```

* Test

    ```python
    python -m learning_to_simulate.train --mode=eval_rollout --eval_split=test --data_path=./learning_to_simulate/datasets/Excavation_PCA --model_path=./learning_to_simulate/models/Excavation_PCA --output_path=./learning_to_simulate/rollouts/Excavation_PCA
    ```

* Visualize

    * 2D plot

        ```python
        python -m learning_to_simulate.render_rollout_2d_force --plane=xy --data_path=./learning_to_simulate/datasets/Excavation_PCA --rollout_path=./learning_to_simulate/rollouts/Excavation_PCA/rollout_test_0.pkl
        ```

    * 3D plot

        ```python
        python -m learning_to_simulate.render_rollout_3d --fullspace=True --data_path=./learning_to_simulate/datasets/Excavation_PCA --rollout_path=./learning_to_simulate/rollouts/Excavation_PCA/rollout_test_0.pkl
        ```


# Bibtex
Please cite our papers
[
[1](https://www.sciencedirect.com/science/article/pii/S0952197624009230),
[2](https://ieeexplore.ieee.org/abstract/document/9438132)
]
if you use this code for your research: 
```
@article{HAERI2024108765,
   title = {Subspace graph networks for real-time granular flow simulation with applications to machine-terrain interactions},
   journal = {Engineering Applications of Artificial Intelligence},
   volume = {135},
   pages = {108765},
   year = {2024},
   issn = {0952-1976},
   doi = {https://doi.org/10.1016/j.engappai.2024.108765},
   url = {https://www.sciencedirect.com/science/article/pii/S0952197624009230},
   author = {Amin Haeri and Daniel Holz and Krzysztof Skonieczny},
   keywords = {Real-time physics simulation, Geometric deep learning, Graph neural networks, Continuum mechanics, Experiment},
}
```
and/or
```
@INPROCEEDINGS{9438132,
    author={Haeri, A. and Skonieczny, K.},
    booktitle={2021 IEEE Aerospace Conference (50100)},
    title={Accurate and Real-time Simulation of Rover Wheel Traction},
    year={2021},
    volume={},
    number={},
    pages={1-9},
    doi={10.1109/AERO50100.2021.9438132}
}
```
