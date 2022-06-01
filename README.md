# Subspace Graph Physics: <br /> Real-Time Rigid Body-Driven Granular Flow Simulation
#### [[Paper 1](https://arxiv.org/abs/2111.10206)] [[Paper 2](https://ieeexplore.ieee.org/abstract/document/9438132)]


This is a <strong>computationally efficient</strong> version of "Learning To Simulate" developed by [DeepMind](https://deepmind.com/research/publications/Learning-to-Simulate-Complex-Physics-with-Graph-Networks) and [Stanford](https://cs.stanford.edu/people/jure) researchers, for <strong>real-time 3D</strong> physics simulations (e.g. for granular flows and their interactions with rigid bodies).

<img src="https://github.com/haeriamin/files/blob/master/excav_ml_4.gif" alt="drawing" width="820">


# Highlights

* We train the graph network (GN) model in subspace by performing Principal Component Analysis (PCA).

* PCA enables GN to be trained using a single desktop GPU with moderate VRAM for large 3D configurations.

* The training datasets can be generated by our efficient and accurate [Material Point Method](https://github.com/haeriamin/MPM-NGF) (MPM).

* The rollout runtime is under 1 sec/sec, and the training runtime is 60 global-step/sec (on NVIDIA RTX 3080).

* The particle positions and velocities, and rigid body interaction forces are compared above.


# Install and Run Demo

* Install

    * Install Python (tested on version 3.7)

    * (Optional) [Install TensorFlow 1.15 for NVIDIA RTX30 GPUs (without docker or CUDA install)](https://www.pugetsystems.com/labs/hpc/How-To-Install-TensorFlow-1-15-for-NVIDIA-RTX30-GPUs-without-docker-or-CUDA-install-2005/)

    * Run

        ```bash
        pip install -r requirements.txt
        ```

* Train:

    ```python
    python3 -m learning_to_simulate.train \
    --mode=train \
    --eval_split=train \
    --batch_size=2 \
    --data_path=./learning_to_simulate/datasets/Excavation_PCA \
    --model_path=./learning_to_simulate/models/Excavation_PCA
    ```

* Test:

    ```python
    python3 -m learning_to_simulate.train \
    --mode=eval_rollout \
    --eval_split=test \
    --data_path=./learning_to_simulate/datasets/Excavation_PCA \
    --model_path=./learning_to_simulate/models/Excavation_PCA \
    --output_path=./learning_to_simulate/rollouts/Excavation_PCA
    ```

* Visualize:

    * 2D plot:

        ```python
        python -m learning_to_simulate.render_rollout_2d_force \
        --plane=xy \
        --data_path=./learning_to_simulate/datasets/Excavation_PCA \
        --rollout_path=./learning_to_simulate/rollouts/Excavation_PCA
        ```

    * 3D plot:

        ```python
        python -m learning_to_simulate.render_rollout_3d_force \
        --fullspace=True \
        --data_path=./learning_to_simulate/datasets/Excavation_PCA \
        --rollout_path=./learning_to_simulate/rollouts/Excavation_PCA/rollout_test_0.pkl
        ```


# Bibtex
Please cite our papers [[1](https://arxiv.org/abs/2111.01523), [2](https://ieeexplore.ieee.org/abstract/document/9438132)] if you use this code for your research: 
```
@misc{haeri2021subspace,
    title={Subspace Graph Physics: Real-Time Rigid Body-Driven Granular Flow Simulation}, 
    author={Amin Haeri and Krzysztof Skonieczny},
    year={2021},
    eprint={2111.10206},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
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
