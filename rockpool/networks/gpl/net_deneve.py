###
# net_deneve.py - Classes and functions for encapsulating Denève reservoirs
###

from ..network import Network
from ...layers import PassThrough, FFExpSyn, FFExpSynBrian
from ...layers import RecFSSpikeEulerBT
from ...timeseries import TSContinuous

import numpy as np

import matplotlib.pyplot as plt

# - Configure exports
__all__ = ["NetworkDeneve"]


class NetworkDeneve(Network):
    """
    `.Network` base class that defines and builds networks of Deneve reservoirs to solve linear problems.
    """

    def __init__(self):
        # - Call super-class constructor
        super().__init__()

    @staticmethod
    def SolveLinearProblem(
        a: np.ndarray = None,
        net_size: int = None,
        gamma: np.ndarray = None,
        dt: float = None,
        mu: float = 1e-4,
        nu: float = 1e-3,
        noise_std: float = 0.0,
        tau_mem: float = 20e-3,
        tau_syn_fast: float = 1e-3,
        tau_syn_slow: float = 100e-3,
        v_thresh: np.ndarray = -55e-3,
        v_rest: np.ndarray = -65e-3,
        refractory=-np.finfo(float).eps,
    ) -> "NetworkDeneve":
        """
        SolveLinearProblem - Static method Direct implementation of a linear dynamical system

        :param a:             np.ndarray [PxP] Matrix defining linear dynamical system
        :param net_size:        int Desired size of recurrent reservoir layer (Default: 100)

        :param gamma:         np.ndarray [PxN] Input kernel (Default: 50 * Normal / net_size)

        :param dt:             float Nominal time step (Default: 0.1 ms)

        :param mu:             float Linear cost parameter (Default: 1e-4)
        :param nu:             float Quadratic cost parameter (Default: 1e-3)

        :param noise_std:       float Noise std. dev. (Default: 0)

        :param tau_mem:           float Neuron membrane time constant (Default: 20 ms)
        :param tau_syn_fast:     float Fast synaptic time constant (Default: 1 ms)
        :param tau_syn_slow:     float Slow synaptic time constant (Default: 100 ms)

        :param v_thresh:       float Threshold membrane potential (Default: -55 mV)
        :param v_rest:         float Rest potential (Default: -65 mV)
        :param refractory: float Refractory time for neuron spiking (Default: 0 ms)

        :return: A configured NetworkDeneve object, containing input, reservoir and output layers
        """
        # - Get input parameters
        nJ = a.shape[0]

        # - Generate random input weights if not provided
        if gamma is None:
            # - Default net size
            if net_size is None:
                net_size = 100

            # - Create random input kernels
            gamma = np.random.randn(nJ, net_size)
            gamma /= np.sum(np.abs(gamma), 0, keepdims=True)
            gamma /= net_size
            gamma *= 50

        else:
            assert (net_size is None) or net_size == gamma.shape[
                1
            ], "`net_size` must match the size of `gamma`."
            net_size = gamma.shape[1]

        # - Generate network
        lambda_d = np.asarray(1 / tau_syn_slow)

        v_t = (
            nu * lambda_d
            + mu * lambda_d ** 2
            + np.sum(abs(gamma.T), -1, keepdims=True) ** 2
        ) / 2

        omega_f = gamma.T @ gamma + mu * lambda_d ** 2 * np.identity(net_size)
        omega_s = gamma.T @ (a + lambda_d * np.identity(nJ)) @ gamma

        # - Scale problem to arbitrary membrane potential ranges and time constants
        t_dash = _expand_to_size(v_thresh, net_size)
        b = np.reshape(_expand_to_size(v_rest, net_size), (net_size, -1))
        a = np.reshape(_expand_to_size(v_thresh, net_size), (net_size, -1)) - b

        gamma_dash = a * gamma.T / v_t

        omega_f_dash = a * omega_f / v_t
        omega_s_dash = a * omega_s / v_t

        # - Pull out reset voltage from fast synaptic weights
        v_reset = t_dash - np.diag(omega_f_dash)
        np.fill_diagonal(omega_f_dash, 0)

        # - Scale omega_f_dash by fast TC
        omega_f_dash /= tau_syn_fast

        # - Scale everything by membrane TC
        omega_f_dash *= tau_mem
        omega_s_dash *= tau_mem
        gamma_dash *= tau_mem

        # - Build and return network
        return NetworkDeneve.SpecifyNetwork(
            weights_fast=-omega_f_dash,
            weights_slow=omega_s_dash,
            weights_in=gamma_dash.T,
            weights_out=gamma.T,
            dt=dt,
            noise_std=noise_std,
            v_rest=v_rest,
            v_reset=v_reset,
            v_thresh=v_thresh,
            tau_mem=tau_mem,
            tau_syn_r_fast=tau_syn_fast,
            tau_syn_r_slow=tau_syn_slow,
            tau_syn_out=tau_syn_slow,
            refractory=refractory,
        )


    def train(self, func, omega_optimal, num_iterations, fCommandAmp, nNumVariables, sigma, tDt, tDuration, verbose = False, validation_step = 10):
        """
        Arguments: self: Network object, access layer attributes self.lyrRes.XX
                   func: Function to generate random input. Takes duration, dt, Amp, Nx, sigma as parameters
                   num_iterations: Number of total training iterations
                   verbose: Track number of updates, decoding error, distance to optimal weights
                   validation_step : Validate on new input after X iterations and produce verbose output
        """
        print("Start training...")
        t = np.linspace(0, tDuration, int(tDuration / tDt))
        def get_ts_input():
            rv = func(duration=tDuration, dt=tDt, Amp=fCommandAmp,Nx=nNumVariables,sigma=sigma)
            return TSContinuous(t,rv.T)

        val_set = [get_ts_input() for _ in range(10)]

        def get_distance(Copt, C):
            optscale = np.trace(np.matmul(C.T, Copt)) / np.sum(Copt**2)
            Cnorm = np.sum(C**2)
            ErrorC = np.sum(np.sum((C - optscale*Copt)**2 ,axis=0)) / Cnorm
            return ErrorC

        def get_decoding_error():
            """
            Computes average error over validation set
            """
            errors = []
            for ts_rv in val_set:
                dResp = self.evolve(ts_rv, tDuration, verbose=False)
                self.reset_all()
                reconstructed = dResp['Output'].samples[:int(tDuration/tDt),:]
                target = ts_rv.samples / fCommandAmp
                scales = np.linalg.pinv(reconstructed.T @ reconstructed) @ reconstructed.T @ target
                recon = scales @ reconstructed.T
                err = np.sum(np.var(target-recon.T, axis=0, ddof=1)) / (np.sum(np.var(target, axis=0, ddof=1)))
                errors.append(err)

            return np.mean(np.asarray(errors))


        iter_num = []
        distance_to_optimal_weights = []
        decoding_error = []

        fig = plt.figure(figsize=(7,5))

        for i in range(num_iterations):

            if(i % validation_step == 0):
                self.lyrRes.is_training = False
                err = get_decoding_error()
                decoding_error.append(err)
                self.lyrRes.is_training = True
                dist = get_distance(omega_optimal, self.lyrRes.weights)
                iter_num.append(i)
                distance_to_optimal_weights.append(dist)
                print("Distance to optimal weights is %.4f" % dist)

            #self.margin = np.log(i + 1)
            #print("Margin is %.4f" % self.margin)

            self.lyrRes.is_training = True
            ts_input = get_ts_input()
            dResp = self.evolve(ts_input, tDuration, verbose=False)
            self.reset_all()

            # Draw
            plt.clf()
            num_neurons = 8 # Plot the voltages and net currents for that many neurons
            net_current = self.lyrRes._last_evolve['f'] # + self.lyrRes._last_evolve['static_input']
            v = self.lyrRes._last_evolve['v']
            times = self.lyrRes._last_evolve['vt']
            c = 1
            base = num_neurons*100+10+0
            for idx in range(num_neurons):
                ax = fig.add_subplot(base+c)
                ax.plot(times, v[idx,:].T)
                c+= 1

            plt.tight_layout()
            plt.draw()
            plt.pause(0.2)


        self.lyrRes.is_training = False
        self.reset_all()

        return (iter_num,distance_to_optimal_weights,decoding_error)



    @staticmethod
    def SpecifyNetwork(
        weights_fast,
        weights_slow,
        weights_in,
        weights_out,
        dt: float = None,
        noise_std: float = 0.0,
        v_thresh: np.ndarray = -55e-3,
        v_reset: np.ndarray = -65e-3,
        v_rest: np.ndarray = -65e-3,
        tau_mem: float = 20e-3,
        tau_syn_r_fast: float = 1e-3,
        tau_syn_r_slow: float = 100e-3,
        tau_syn_out: float = 100e-3,
        refractory: float = -np.finfo(float).eps,
        granularity: float = 1,
        margin: float = 1,
    ):
        """
        SpecifyNetwork - Directly configure all layers of a reservoir

        :param weights_fast:       np.ndarray [NxN] Matrix of fast synaptic weights
        :param weights_slow:       np.ndarray [NxN] Matrix of slow synaptic weights
        :param weights_in:   np.ndarray [LxN] Matrix of input kernels
        :param weights_out:  np.ndarray [NxM] Matrix of output kernels

        :param dt:         float Nominal time step
        :param noise_std:   float Noise Std. Dev.

        :param v_rest:     np.ndarray [Nx1] Vector of rest potentials (spiking layer)
        :param v_reset:    np.ndarray [Nx1] Vector of reset potentials (spiking layer)
        :param v_thresh:   np.ndarray [Nx1] Vector of threshold potentials (spiking layer)

        :param tau_mem:      float Neuron membrane time constant (spiking layer)
        :param tau_syn_r_fast: float Fast recurrent synaptic time constant
        :param tau_syn_r_slow: float Slow recurrent synaptic time constant
        :param tau_syn_out:    float Synaptic time constant for output layer

        :param refractory: float Refractory time for spiking layer

        :return:
        """

        def spike_callback(layer, time, spike_id, v_last, verbose=False):
            """
            Implements the discrete learning rule.
            Condition for incrementing fOmega{n,k} by w : 2*v_last{n} + fOmega{n,k} < -w/2 - margin
            Condition for decrementing fOmega{n,k} by w : 2*v_last{n} + fOmega{n,k} >  w/2 + margin

            Incrementing or decrementing the fOmega{i,i} corresponds to incrementing or decrementing the
            reset potential.
            The diagonal of fOmega is always zero and the relation V_reset = V_thresh + diagonal(Omega) holds,
            where the thresholds are fixed throughout training and the weights are assumed to be -(FF.T + mu*I)
            """

            k = spike_id
            num_updates = 0
            if(verbose and k == layer.neuron_k):
                layer.spike_times_n_k.append(time)
                layer.v_n_before.append(v_last[layer.neuron_n])
                layer.v_n_after.append(layer._state[layer.neuron_n])

            for n in range(layer.size):
                # Check for connection Omega{n,k}
                if(n == k):
                    omega_n_k = layer.v_reset[n] + layer.v_thresh[n]
                else:
                    omega_n_k = layer._weights[n,k] * tau_mem

                if(2*v_last[n] + omega_n_k < -layer.granularity/2 + 2*layer.v_rest[n]):
                    num_updates += 1
                    if(n == k):
                        # layer.v_reset[n] += 0.1
                        # layer.v_rest[n] += 0.1
                        pass
                    else:
                        layer._weights[n,k] += layer.granularity
                        if(verbose and k == layer.neuron_k and n == layer.neuron_n):
                            layer.correction_times.append(time)
                            
                elif(2*v_last[n] + omega_n_k > layer.granularity/2 + 2*layer.v_rest[n]):
                    num_updates += 1
                    if(n == k):
                        # layer.v_reset[n] -= 0.1
                        # layer.v_rest[n] -= 0.1
                        pass
                    else:
                        layer._weights[n,k] -= layer.granularity
                        if(verbose and k == layer.neuron_k and n == layer.neuron_n):
                            layer.correction_times.append(time)

            return num_updates

        # - Construct reservoir
        reservoir_layer = RecFSSpikeEulerBT(
            weights_fast,
            weights_slow,
            dt=dt,
            noise_std=noise_std,
            tau_mem=tau_mem,
            tau_syn_r_fast=tau_syn_r_fast,
            tau_syn_r_slow=tau_syn_r_slow,
            v_thresh=v_thresh,
            v_rest=v_rest,
            v_reset=v_reset,
            refractory=refractory,
            spike_callback=spike_callback,
            granularity=granularity,
            margin=margin,
            name="DeneveReservoir",
        )

        # - Ensure time step is consistent across layers
        if dt is None:
            dt = reservoir_layer.dt

        # - Construct input layer
        input_layer = PassThrough(weights_in, dt=dt, noise_std=noise_std, name="Input")

        # - Construct output layer
        output_layer = FFExpSyn(
            weights_out,
            dt=dt,
            noise_std=noise_std,
            tau_syn=tau_syn_out,
            name="Output",
        )

        # - Build network
        net_deneve = NetworkDeneve()
        net_deneve.input_layer = net_deneve.add_layer(input_layer, external_input=True)
        net_deneve.lyrRes = net_deneve.add_layer(reservoir_layer, input_layer)
        net_deneve.output_layer = net_deneve.add_layer(output_layer, reservoir_layer)

        # - Return constructed network
        return net_deneve


def _expand_to_size(inp, size: int, var_name: str = "input") -> np.ndarray:
    """
    _expand_to_size: Replicate out a scalar to a desired size

    :param inp:          scalar or array-like (N)
    :param size:           int Desired size to return (=N)
    :param var_name:   str Name of the variable to include in error messages
    :return:                np.ndarray (N) vector
    """
    if np.size(inp) == 1:
        # - Expand input to vector
        inp = np.repeat(inp, size)

    assert np.size(inp) == size, "`{}` must be a scalar or have {} elements".format(
        var_name, size
    )

    # - Return object of correct shape
    return np.reshape(inp, size)
