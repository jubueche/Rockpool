"""
Implement the layer for the NetworkADS (Arbitrary Dynamical System), which is capable of learning
an arbitrary dynamical system.
"""

from ..layer import Layer
from rockpool.timeseries import TSEvent, TSContinuous, TimeSeries, get_global_ts_plotting_backend
import numpy as np
from typing import Union, Callable, Any, Tuple, Optional
import copy
from numba import njit
from warnings import warn

# - Try to import holoviews
try:
    import holoviews as hv
except Exception:
    pass

import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
import matplotlib.pyplot as plt # For quick plottings

__all__ = ["RecFSSpikeADS"]


@njit
def _backstep(vCurrent, vLast, tStep, tDesiredStep):
    return (vCurrent - vLast) / tStep * tDesiredStep + vLast

# @njit
def neuron_dot_v(
    t,
    V,
    dt,
    I_s_F,
    I_W_slow_phi_x,
    I_kDte,
    I_ext,
    V_rest,
    tau_V,
    bias
):
    return (V_rest - V + I_s_F + I_W_slow_phi_x + I_kDte + I_ext + bias) / tau_V

@njit
def syn_dot_I(t, I, dt, I_spike, tau_Syn):
    return -I / tau_Syn + I_spike / dt

class RecFSSpikeADS(Layer):
    """
    Implement the layer for the NetworkADS (Arbitrary Dynamical System), which is capable of learning
    an arbitrary dynamical system.
    See rockpool/networks/gpl/net_as.py for the parameters passed here.
    """
    def __init__(self,
                weights_fast : np.ndarray,
                weights_slow : np.ndarray,
                weights_in : np.ndarray,
                weights_out : np.ndarray,
                M : np.ndarray,
                theta : np.ndarray,
                eta : float,
                k : float,
                bias: np.ndarray,
                noise_std : float,
                dt : float,
                v_thresh : Union[np.ndarray,float],
                v_reset : Union[np.ndarray,float],
                v_rest : Union[np.ndarray,float],
                tau_mem : float,
                tau_syn_r_fast : float,
                tau_syn_r_slow : float,
                refractory : float,
                phi : Callable[[np.ndarray],np.ndarray],
                learning_callback,
                record : bool,
                name : str):
        
        super().__init__(weights=weights_fast, noise_std=noise_std, name=name)

        # Fast weights, noise_std and name are access. via self.XX or self._XX
        self.weights_slow = weights_slow
        self.weights_out = weights_out
        self.weights_in = weights_in
        self.M = M
        self.theta = theta
        self.eta = eta
        self.k = k
        self.bias = np.asarray(bias).astype("float")
        self.v_thresh = np.asarray(v_thresh).astype("float")
        self.v_reset = np.asarray(v_reset).astype("float")
        self.v_rest = np.asarray(v_rest).astype("float")
        self.tau_mem = np.asarray(tau_mem).astype("float")
        self.tau_syn_r_fast = np.asarray(tau_syn_r_fast).astype("float")
        self.tau_syn_r_slow = np.asarray(tau_syn_r_slow).astype("float")
        self.refractory = float(refractory)
        self.phi = phi
        self.is_training = False # Set externally by train method in net_ads.py
        self._ts_target = None
        self.recorded_states = None
        self.record = record
        self.k_initial = k
        self.eta_initial = eta

        self.learning_callback = learning_callback


        # - Set a reasonable dt
        if dt is None:
            self.dt = self._min_tau / 10
        else:
            self.dt = np.asarray(dt).astype("float")

        # Initialise the network
        self.reset_all()

    def reset_state(self):
        """
        Reset the internal state of the network
        """
        self.state = self.v_rest.copy()
        self.I_s_S = np.zeros(self.size)
        self.I_s_F = np.zeros(self.size)


    def evolve(self,
                ts_input: Optional[TSContinuous] = None,
                duration: Optional[float] = None,
                num_timesteps: Optional[int] = None,
                verbose: bool = False,
                min_delta: Optional[float] = None,) -> TSEvent:
        """
        Evolve the function on the input c(t). This function simply feeds the input through the network and does not perform any learning

        Params:
            ts_input : [TSContinuous] Corresponds to the input c in the simulations, shape: [int(duration/dt), N], eg [30001,100]
            duration : [float] Duration in seconds for the layer to be evolved, net_ads calls evolve with duration set to None, but passes num_timesteps
            num_timesteps : [int] Number of timesteps to be performed. Typically int(duration / dt) where duration is passed to net.evolve (not the layer evolve)
            verbose : [bool] Print verbose output
            min_delta : [float] Minimal time set taken. Typically 1/10*dt . Must be strictly smaller than dt. This is used to determine the precise spike timing  
        """

        # - Work out reasonable default for nominal time step (1/10 fastest time constant)
        if min_delta is None:
            min_delta = self.dt / 10

        # - Check time step values
        assert min_delta < self.dt, "`min_delta` must be shorter than `dt`"

        # - Get discretised input and nominal time trace
        input_time_trace, static_input, num_timesteps = self._prepare_input(ts_input, duration, num_timesteps)
        final_time = input_time_trace[-1]

        # Assertions about training and the targets that were set
        if(self.is_training):
            assert (self.learning_callback is not None), "Learning flag set, but no callback provided"
            assert (self.ts_target is not None), "Evolve called with learning flag set, but no target input provided"
            assert (input_time_trace.shape == self.target_time_trace.shape), "Input and target time_trace shapes don't match"
            assert (static_input.shape[0] == self.static_target.shape[0]), "Input and target lengths don't match"
            assert (num_timesteps == self.target_num_timesteps), "Input and output num_timesteps don't match"

        # - Generate a noise trace
        noise_step = (
            np.random.randn(np.size(input_time_trace), self.size) * self.noise_std
        )
        static_input += noise_step

        # - Allocate state storage variables
        record_length = num_timesteps
        spike_pointer = 0
        times = full_nan(record_length)
        v = full_nan((self.size, record_length))
        s = full_nan((self.size, record_length))
        f = full_nan((self.size, record_length))
        I_kDte_track = full_nan((self.size, record_length))
        I_W_slow_phi_x_track = full_nan((self.size, record_length))
        dot_v_ts = full_nan((self.size, record_length))

        # - Allocate storage for spike times
        max_spike_pointer = record_length * self.size
        spike_times = full_nan(max_spike_pointer)
        spike_indices = full_nan(max_spike_pointer)

        # - Refractory time variable
        vec_refractory = np.zeros(self.size)

        # - Initialise step and "previous step" variables
        t_time = self._t
        t_start = self._t
        step = 0
        t_last = 0.0
        v_last = self._state.copy()
        I_s_S_Last = self.I_s_S.copy()
        I_s_F_Last = self.I_s_F.copy()

        zeros = np.zeros(self.size)

        # - Euler integrator loop
        while t_time < final_time:

            ### --- Numba-compiled inner function for speed
            # @njit
            def _evolve_backstep(
                t_time,
                weights,
                weights_slow,
                weights_in,
                weights_out,
                M,
                theta,
                phi,
                k,
                is_training,
                state,
                I_s_S,
                I_s_F,
                dt,
                v_last,
                I_s_S_Last,
                I_s_F_Last,
                v_reset,
                v_rest,
                v_thresh,
                bias,
                tau_mem,
                tau_syn_r_slow,
                tau_syn_r_fast,
                refractory,
                vec_refractory,
                zeros,
            ):

                # - Enforce refractory period by clamping membrane potential to reset
                state[vec_refractory > 0] = v_reset[vec_refractory > 0]

                ## - Back-tick spike detector

                # - Locate spiking neurons
                spike_ids = state > v_thresh
                spike_ids = argwhere(spike_ids)
                num_spikes = len(spike_ids)

                # - Were there any spikes?
                if num_spikes > 0:
                    # - Predict the precise spike times using linear interpolation, returns a value between 0 and dt depending on whether V(t) or V(t-1) is closer to the threshold.
                    # We then have the precise spike time by adding spike_delta to the last time instance. If it is for example 0, we add the minimal time step, namely min_delta
                    spike_deltas = (
                        (v_thresh[spike_ids] - v_last[spike_ids])
                        * dt
                        / (state[spike_ids] - v_last[spike_ids])
                    )

                    # - Was there more than one neuron above threshold?
                    if num_spikes > 1:
                        # - Find the earliest spike
                        spike_delta, first_spike_id = min_argmin(spike_deltas)
                        first_spike_id = spike_ids[first_spike_id]
                    else:
                        spike_delta = spike_deltas[0]
                        first_spike_id = spike_ids[0]

                    # - Find time of actual spike
                    shortest_step = t_last + min_delta
                    spike = clip_scalar(t_last + spike_delta, shortest_step, t_time)
                    spike_delta = spike - t_last

                    # - Back-step time to spike
                    t_time = spike
                    vec_refractory = vec_refractory + dt - spike_delta

                    # - Back-step all membrane and synaptic potentials to time of spike (linear interpolation)
                    state = _backstep(state, v_last, dt, spike_delta)
                    I_s_S = _backstep(I_s_S, I_s_S_Last, dt, spike_delta)
                    I_s_F = _backstep(I_s_F, I_s_F_Last, dt, spike_delta)

                    # - Apply reset to spiking neuron
                    state[first_spike_id] = v_reset[first_spike_id]

                    # - Begin refractory period for spiking neuron
                    vec_refractory[first_spike_id] = refractory

                    # - Set spike currents
                    I_spike_slow = np.zeros(self.size)
                    I_spike_slow[first_spike_id] = 1.0
                    I_spike_fast = weights[:, first_spike_id]

                else:
                    # - Clear spike currents
                    first_spike_id = -1
                    I_spike_slow = zeros
                    I_spike_fast = zeros

                ### End of back-tick spike detector
                # - Save synapse and neuron states for previous time step
                v_last[:] = state
                I_s_S_Last[:] = I_s_S + I_spike_slow
                I_s_F_Last[:] = I_s_F + I_spike_fast

                # - Update synapse and neuron states (Euler step)
                dot_I_s_S = syn_dot_I(t_time, I_s_S, dt, I_spike_slow, tau_syn_r_slow)
                I_s_S += dot_I_s_S * dt

                dot_I_s_F = syn_dot_I(t_time, I_s_F, dt, I_spike_fast, tau_syn_r_fast)
                I_s_F += dot_I_s_F * dt

                # Calculate phi(M@r +theta)
                phi_r = phi(M @ I_s_S + theta)
                I_W_slow_phi_x = weights_slow.T @ phi_r

                int_time = int((t_time - t_start) // dt)
                I_ext = static_input[int_time, :]

                if(is_training):
                    # Compute x_hat
                    x = self.static_target[int_time, :]
                    x_hat = weights_out.T @ I_s_S
                    e = x - x_hat
                    I_kDte = k*weights_in.T @ e
                else:
                    I_kDte = zeros
                    assert (I_kDte == 0).all(), "I_kDte is not zero"
                    e = 0

                # Calculate dot_v following dot_v = -lam + I_ext + I_s_F + I_W_slow_phi_x + I_kDte,  (V_rest - V + I_s_F + I_W_slow_phi_x + I_kDte + I_ext + bias) / tau_V
                dot_v = neuron_dot_v(
                    t = t_time,
                    V = state,
                    dt = dt,
                    I_s_F = I_s_F,
                    I_W_slow_phi_x = I_W_slow_phi_x,
                    I_kDte = I_kDte,
                    I_ext = I_ext,
                    V_rest = v_rest,
                    tau_V = tau_mem,
                    bias = bias
                )

                state += dot_v * dt

                return (
                    t_time,
                    first_spike_id,
                    dot_v,
                    state,
                    I_s_S,
                    I_s_F,
                    I_W_slow_phi_x,
                    I_kDte,
                    dt,
                    v_last,
                    I_s_S_Last,
                    I_s_F_Last,
                    vec_refractory,
                    phi_r,
                    e
                )

            ### --- END of compiled inner function
            (
                t_time,
                first_spike_id,
                dot_v,
                self._state,
                self.I_s_S,
                self.I_s_F,
                I_W_slow_phi_x,
                I_kDte,
                self._dt,
                v_last,
                I_s_S_Last,
                I_s_F_Last,
                vec_refractory,
                phi_r,
                e,
            ) = _evolve_backstep(
                t_time=t_time,
                weights= self._weights,
                weights_slow=self.weights_slow,
                weights_in=self.weights_in,
                weights_out = self.weights_out,
                M = self.M,
                theta = self.theta,
                phi = self.phi,
                k = self.k,
                is_training = self.is_training,
                state = self._state,
                I_s_S = self.I_s_S,
                I_s_F = self.I_s_F,
                dt = self._dt,
                v_last = v_last,
                I_s_S_Last = I_s_S_Last,
                I_s_F_Last = I_s_F_Last,
                v_reset = self.v_reset,
                v_rest = self.v_rest,
                v_thresh = self.v_thresh,
                bias = self.bias,
                tau_mem = self.tau_mem,
                tau_syn_r_slow = self.tau_syn_r_slow,
                tau_syn_r_fast = self.tau_syn_r_fast,
                refractory = self.refractory,
                vec_refractory = vec_refractory,
                zeros = zeros
            )

            # Call the training. Note this is not spike based (but should probably be). Add "first_spike_id > -1 and" to if to make spike based
            if(self.is_training and self.learning_callback is not None):
                dot_W_slow = self.learning_callback(weights_slow = self.weights_slow, phi_r=phi_r, weights_in=self.weights_in, e = e, dt=self.dt)  # def l(W_slow, eta, phi_r, weights_in, e, dt):
                self.weights_slow = self.weights_slow + self.eta*dot_W_slow

            # - Extend spike record, if necessary
            if spike_pointer >= max_spike_pointer:
                extend = int(max_spike_pointer // 2)
                spike_times = np.append(spike_times, full_nan(extend))
                spike_indices = np.append(spike_indices, full_nan(extend))
                max_spike_pointer += extend

            # - Record spiking neuron
            spike_times[spike_pointer] = t_time
            spike_indices[spike_pointer] = first_spike_id
            spike_pointer += 1

            # - Extend state storage variables, if needed
            if step >= record_length:
                extend = num_timesteps
                times = np.append(times, full_nan(extend))
                v = np.append(v, full_nan((self.size, extend)), axis=1)
                s = np.append(s, full_nan((self.size, extend)), axis=1)
                f = np.append(f, full_nan((self.size, extend)), axis=1)
                I_kDte_track = np.append(I_kDte_track, full_nan((self.size, extend)), axis=1)
                I_W_slow_phi_x_track = np.append(I_W_slow_phi_x_track, full_nan((self.size, extend)), axis=1)
                dot_v_ts = np.append(dot_v_ts, full_nan((self.size, extend)), axis=1)
                record_length += extend

             # - Store the network states for this time step
            times[step] = t_time
            v[:, step] = self._state
            s[:, step] = self.I_s_S
            f[:, step] = self.I_s_F
            I_kDte_track[:, step] = I_kDte
            I_W_slow_phi_x_track[:, step] = I_W_slow_phi_x
            dot_v_ts[:, step] = dot_v

            # - Next nominal time step
            t_last = copy.copy(t_time)
            t_time += self._dt
            step += 1
            vec_refractory -= self.dt
        ## END End of Euler integration loop

        ## - Back-step to exact final time
        self.state = _backstep(self.state, v_last, self._dt, t_time - final_time)
        self.I_s_S = _backstep(self.I_s_S, I_s_S_Last, self._dt, t_time - final_time)
        self.I_s_F = _backstep(self.I_s_F, I_s_F_Last, self._dt, t_time - final_time)

        ## - Store the network states for final time step
        times[step - 1] = final_time
        v[:, step - 1] = self.state
        s[:, step - 1] = self.I_s_S
        f[:, step - 1] = self.I_s_F
        I_kDte_track[:, step - 1] = I_kDte
        I_W_slow_phi_x_track[:, step - 1] = I_W_slow_phi_x

        ## - Trim state storage variables
        times = times[:step]
        v = v[:, :step]
        s = s[:, :step]
        f = f[:, :step]
        I_kDte_track = I_kDte_track[:, :step]
        I_W_slow_phi_x_track = I_W_slow_phi_x_track[:, :step]
        dot_v_ts = dot_v_ts[:, :step]
        spike_times = spike_times[:spike_pointer]
        spike_indices = spike_indices[:spike_pointer]

        ## - Construct return time series
        resp = {
            "vt": times,
            "mfX": v,
            "s": s,
            "f": f,
            "I_kDte" : I_kDte_track,
            "I_W_slow_phi_x" : I_W_slow_phi_x_track,
            "mfFast": f,
            "dot_v": dot_v_ts,
            "static_input": static_input,
        }

        backend = get_global_ts_plotting_backend()
        if backend is "holoviews":
            spikes = {"times": spike_times, "vnNeuron": spike_indices}

            resp["spReservoir"] = hv.Points(
                spikes, kdims=["times", "vnNeuron"], label="Reservoir spikes"
            ).redim.range(times=(0, num_timesteps * self.dt), vnNeuron=(0, self.size))
        else:
            resp["spReservoir"] = dict(times=spike_times, vnNeuron=spike_indices)

        # - Convert some elements to time series
        resp["tsX"] = TSContinuous(resp["vt"], resp["mfX"].T, name="Membrane potential")
        resp["tsA"] = TSContinuous(resp["vt"], resp["s"].T, name="Slow synaptic state")

        # - Store "last evolution" state
        self._last_evolve = resp
        self._timestep += num_timesteps

        if(self.record):
            self.recorded_states = resp

        # - Return output TimeSeries
        ts_event_return = TSEvent(spike_times, spike_indices)
        if(verbose):
            
            # Compare the current traces
            fig = plt.figure(figsize=(20,20))
            ax0 = fig.add_subplot(511)
            ax0.plot(times, f[0:5,:].T)
            ax0.set_title(r"$I_f$")

            ax1 = fig.add_subplot(512)
            ax1.plot(times, dot_v_ts[0:5,:].T)
            ax1.set_title(r"$\dot{v}(t)$")

            ax2 = fig.add_subplot(513)
            ax2.plot(times, I_W_slow_phi_x_track[0:5,:].T)
            ax2.set_title(r"$I_{W_{slow}^T\phi(r)}$")

            ax3 = fig.add_subplot(514)
            ax3.plot(np.linspace(0,len(static_input[:,0])/self.dt,len(static_input[:,0])), static_input[:,0:5])
            ax3.set_title(r"$I_{ext}$")
            
            _ = fig.add_subplot(515)
            ts_event_return.plot()
            plt.xlim([0,3])
                
            plt.tight_layout()
            plt.show()

            im = plt.matshow(self.weights_slow, fignum=False)
            plt.xticks([], [])
            plt.colorbar(im,fraction=0.046, pad=0.04)
            plt.show()
                

        return ts_event_return

    def to_dict(self):
        NotImplemented

    @property
    def _min_tau(self):
        """
        (float) Smallest time constant of the layer
        """
        return min(np.min(self.tau_syn_r_slow), np.min(self.tau_syn_r_fast))

    @property
    def output_type(self):
        """ (`TSEvent`) Output `TimeSeries` class (`TSEvent`) """
        return TSEvent


    @property
    def tau_syn_r_f(self):
        """ (float) Fast synaptic time constant (s) """
        return self.__tau_syn_r_f

    @tau_syn_r_f.setter
    def tau_syn_r_f(self, tau_syn_r_f):
        self.__tau_syn_r_f = self._expand_to_net_size(tau_syn_r_f, "tau_syn_r_f")

    @property
    def tau_syn_r_s(self):
        """ (float) Slow synaptic time constant (s) """
        return self.__tau_syn_r_s

    @tau_syn_r_s.setter
    def tau_syn_r_s(self, tau_syn_r_s):
        self.__tau_syn_r_s = self._expand_to_net_size(tau_syn_r_s, "tau_syn_r_s")

    @property
    def v_thresh(self):
        """ (float) Threshold potential """
        return self.__thresh

    @v_thresh.setter
    def v_thresh(self, v_thresh):
        self.__thresh = self._expand_to_net_size(v_thresh, "v_thresh")

    @property
    def v_rest(self):
        """ (float) Resting potential """
        return self.__rest

    @v_rest.setter
    def v_rest(self, v_rest):
        self.__rest = self._expand_to_net_size(v_rest, "v_rest")

    @property
    def v_reset(self):
        """ (float) Reset potential"""
        return self.__reset

    @v_reset.setter
    def v_reset(self, v_reset):
        self.__reset = self._expand_to_net_size(v_reset, "v_reset")

    @property
    def bias(self):
        """ (float) Bias potential"""
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = self._expand_to_net_size(bias, "bias")

    @Layer.dt.setter
    def dt(self, new_dt):
        assert (
            new_dt <= self._min_tau / 10
        ), "`new_dt` must be shorter than 1/10 of the shortest time constant, for numerical stability."

        # - Call super-class setter
        super(RecFSSpikeADS, RecFSSpikeADS).dt.__set__(self, new_dt)

    @property
    def theta(self):
        return self._theta
    
    @theta.setter
    def theta(self, t):
        if(t.ndim > 1):
            self._theta = t.ravel()
        else:
            self._theta = t
        assert self._theta.shape[0] == self.M.shape[0], ("Theta must have shape (%d,)" % self.M.shape[0])

    @property
    def ts_target(self):
        return self._ts_target

    @ts_target.setter
    def ts_target(self, t):
        if(t is not None):
            self.target_time_trace, self.static_target, self.target_num_timesteps = self._prepare_input(t, is_target=True)
            self._ts_target = t
        else:
            self._ts_target = None

        

    # Override _prepare_input to also allow preparing the target variable and setting the time base according to the layers dt
    def _prepare_input(
        self,
        ts_input: Optional[TimeSeries] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        is_target: Optional[bool] = False,
    ) -> (np.ndarray, np.ndarray, float):
        """
        Sample input, set up time base

        This function checks an input signal, and prepares a discretised time base according to the time step of the current layer

        :param Optional[TimeSeries] ts_input:   :py:class:`TimeSeries` of TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds. If not provided, then either ``num_timesteps`` or the duration of ``ts_input`` will define the evolution time
        :param Optional[int] num_timesteps:     Integer number of evolution time steps, in units of ``.dt``. If not provided, then ``duration`` or the duration of ``ts_input`` will define the evolution time

        :return (ndarray, ndarray, float): (time_base, input_steps, duration)
            time_base:      T1 Discretised time base for evolution
            input_steps:    (T1xN) Discretised input signal for layer
            num_timesteps:  Actual number of evolution time steps, in units of ``.dt``
        """

        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Generate discrete time base
        time_base = self._gen_time_trace(self.t, num_timesteps)

        if ts_input is not None:
            # - Make sure time_base matches ts_input
            if not isinstance(ts_input, TSEvent):
                if not ts_input.periodic:
                    # - If time base limits are very slightly beyond ts_input.t_start and ts_input.t_stop, match them
                    if (
                        ts_input.t_start - 1e-3 * self.dt
                        <= time_base[0]
                        <= ts_input.t_start
                    ):
                        time_base[0] = ts_input.t_start
                    if (
                        ts_input.t_stop
                        <= time_base[-1]
                        <= ts_input.t_stop + 1e-3 * self.dt
                    ):
                        time_base[-1] = ts_input.t_stop

                # - Warn if evolution period is not fully contained in ts_input
                if not (ts_input.contains(time_base) or ts_input.periodic):
                    warn(
                        "Layer `{}`: Evolution period (t = {} to {}) ".format(
                            self.name, time_base[0], time_base[-1]
                        )
                        + "is not fully contained in input signal (t = {} to {}).".format(
                            ts_input.t_start, ts_input.t_stop
                        )
                        + " You may need to use a `periodic` time series."
                    )

            if(not is_target):
                # - Sample input trace and check for correct dimensions
                input_steps = self._check_input_dims(ts_input(time_base))
            else:
                input_steps = ts_input(time_base)
                if input_steps.ndim == 1 or (input_steps.ndim > 1 and input_steps.shape[1]) == 1:
                    if self.size_in > 1:
                        warn(
                            f"Layer `{self.name}`: Only one channel provided in input - will "
                            + f"be copied to all {self.size_in} input channels."
                        )
                    input_steps = np.repeat(input_steps.reshape((-1, 1)), self._size_in, axis=1)

            # - Treat "NaN" as zero inputs
            input_steps[np.where(np.isnan(input_steps))] = 0

        else:
            # - Assume zero inputs
            input_steps = np.zeros((np.size(time_base), self.size_in))

        return time_base, input_steps, num_timesteps


###### Convenience functions

# - Convenience method to return a nan array
def full_nan(shape: Union[tuple, int]):
    a = np.empty(shape)
    a.fill(np.nan)
    return a


### --- Compiled concenience functions


@njit
def min_argmin(data: np.ndarray) -> Tuple[float, int]:
    """
    Accelerated function to find minimum and location of minimum

    :param data:  np.ndarray of data

    :return (float, int):        min_val, min_loc
    """
    n = 0
    min_loc = -1
    min_val = np.inf
    for x in data:
        if x < min_val:
            min_loc = n
            min_val = x
        n += 1

    return min_val, min_loc


@njit
def argwhere(data: np.ndarray) -> list:
    """
    Accelerated argwhere function

    :param np.ndarray data:  Boolean array

    :return list:         vnLocations where data = True
    """
    vnLocs = []
    n = 0
    for val in data:
        if val:
            vnLocs.append(n)
        n += 1

    return vnLocs


@njit
def clip_vector(v: np.ndarray, f_min: float, f_max: float) -> np.ndarray:
    """
    Accelerated vector clip function

    :param np.ndarray v:
    :param float min:
    :param float max:

    :return np.ndarray: Clipped vector
    """
    v[v < f_min] = f_min
    v[v > f_max] = f_max
    return v


@njit
def clip_scalar(val: float, f_min: float, f_max: float) -> float:
    """
    Accelerated scalar clip function

    :param float val:
    :param float min:
    :param float max:

    :return float: Clipped value
    """
    if val < f_min:
        return f_min
    elif val > f_max:
        return f_max
    else:
        return val


def rep_to_net_size(data: Any, size: Tuple):
    """
    Repeat some data to match the layer size

    :param Any data:
    :param Tuple size:

    :return np.ndarray:
    """
    if np.size(data) == 1:
        return np.repeat(data, size)
    else:
        return data.flatten()