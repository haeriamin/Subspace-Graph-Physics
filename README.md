# Subspace Graph Physics: <br /> Real-Time Rigid Body-Driven Granular Flow Simulation

This is a computationally efficient version of "Learning To Simulate" developed by [DeepMind and Stanford](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate) for real-time physics simulation, specially granular flows and their interactions with rigid bodies. The approach is developed under the supervision of [Prof. Krzysztof Skonieczny (Concordia)](http://users.encs.concordia.ca/~kskoniec/).

<img src="https://github.com/haeriamin/files/blob/master/excav_ml_1.gif" alt="drawing" width="820">
<img src="https://github.com/haeriamin/files/blob/master/excav_ml_4.gif" alt="drawing" width="820">


<!-- ## Code structure

* `run.py`: Runs the optimization.

    * The initial, lower and upper bounds of optimization variables are defined here.
        ```python
        X0 = [,]
        LB = [,]
        UB = [,]
        ```

    * Some other settings including loading/saving optimal solution, and excavation depth and time can also be set.
        ```python
        load = True/False
        save = True/False
        depths = [,]  # [m]
        sim_times = [,]  # [sec]
        ```


* `constr_nm.py`: Implements the constrained Nelder-Mead method [(reference)](https://github.com/alexblaessle/constrNMPy).

* `nelder_mead.py`: Implements the Nelder-Mead method [(reference)](https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py).

    * This is modified to terminate the optimization loop when no significant error changes happen (e.g. `<1`%) during the last specified iterations by setting e.g. `history = 10` as fallows:

        ```python
        if iterations > history+2:
            for i in range(2,history+2):
                fval_sum += abs(fval_history[-1] - fval_history[-i])
            if fval_sum/history < 1:
                break
        ```

* `obj_func.py`: Implements the objective function.

    * The Vortex (excavation) model is called here and implemented in:

        ```python
        def run_vortex(self, x, depth):
            ...
        ```

    * The mean absolute percentage error (MAPE) is calculated using the results from Vortex and experiment.

    * The Vortex files and reference (experimental) results should already be provided in folder `input/`.


## Requirements

* VxSim
* numpy
* pickle -->
