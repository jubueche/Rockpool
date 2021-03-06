##
# Firing rate reservoir with jax back-end
#
# Includes `RecRateEulerJax` - jax-backed firing rate reservoir
# Includes `RecRateEulerJax_IO` - jax-backed firing rate reservoir with input / output weighting
# Includes `ForceRateEulerJax_IO` - jax-backed firing rate FORCE reservoir with input / output weighting
# Includes `FFRateEulerJax` - jax-backed firing rate feed-forward layer
#
# `RecRateEulerJax` and `RecRateEulerJax_IO` are standard reservoirs, with input, recurrent and output layers
# `ForceRateEulerJax` is a driven reservoir, where "recurrent" inputs are inserted from an external source.
#       Used in reservoir transfer.
##

# -- Imports
import jax.numpy as np
from jax import jit
from jax.lax import scan
import jax.random as rand
import numpy as onp
from typing import Optional, Tuple, Callable, Union
from warnings import warn

FloatVector = Union[float, np.ndarray]

from ..layer import Layer
from ...timeseries import TimeSeries, TSContinuous


# -- Define module exports
__all__ = ["RecRateEulerJax", "RecRateEulerJax_IO", "ForceRateEulerJax_IO", "FFRateEulerJax","H_ReLU", "H_tanh"]


# -- Define useful neuron transfer functions
def H_ReLU(x: FloatVector) -> FloatVector:
    return np.clip(x, 0, None)

def H_tanh(x: FloatVector) -> FloatVector:
    return np.tanh(x)


# -- Generators for compiled evolution functions


def _get_rec_evolve_jit(H: Callable[[float], float]):
    """
    _get_rec_evolve_jit() - Return a compiled raw reservoir evolution function

    :param H:   Callable[[float], float] Neuron activation function
    :return:     f(x0, w_in, w_recurrent, w_out, bias, tau, inputs, noise_std, key, dt) -> (x, res_inputs, rec_inputs, res_acts, outputs)
    """

    @jit
    def rec_evolve_jit(
        x0: np.ndarray,
        w_in: FloatVector,
        w_recurrent: FloatVector,
        w_out: FloatVector,
        bias: FloatVector,
        tau: FloatVector,
        inputs: np.ndarray,
        noise_std: float,
        key,
        dt: float,
    ):
        """
        rec_evolve_jit() - Compiled recurrent evolution function

        :param x0:          np.ndarray Initial state of reservoir units
        :param w_in:        np.ndarray Input weights [IxN]
        :param w_recurrent: np.ndarray Recurrent weights [NxN]
        :param w_out:       np.ndarray Output weights [NxO]
        :param bias:        np.ndarray Bias values of reservoir units [N]
        :param tau:         np.ndarray Time constants of reservoir units [N]
        :param inputs:      np.ndarray Input time series [TxN]
        :param noise_std:   float Standard deviation of noise injected into reservoir units
        :param key:         Jax RNG key to use in noise generation
        :param dt:          float Time step for forward Euler solver

        :return:    (x, res_inputs, rec_inputs, res_acts, outputs)
                x:          np.ndarray State of
                res_inputs: np.ndarray Time series of weighted external inputs to each reservoir unit [TxN]
                rec_inputs: np.ndarray Time series of recurrent inputs to each reservoir unit [TxN]
                res_acts:   np.ndarray Time series of reservoir unit activities [TxN]
                outputs:    np.ndarray Time series of output layer values [TxO]
        """
        # - Pre-compute dt/tau
        dt_tau = dt / tau

        # - Reservoir state step function (forward Euler solver)
        def reservoir_step(x, inps):
            """
            reservoir_step() - Single step of recurrent reservoir

            :param x:       np.ndarray Current state of reservoir units
            :param inps:    np.ndarray Inputs to each reservoir unit for the current step

            :return:    xnext, (rec_input, activation)
            """
            inp, rand = inps
            activation = H(x)
            rec_input = np.dot(activation, w_recurrent)
            dx = dt_tau * (-x + inp + bias + rand + rec_input)

            return x + dx, (rec_input, activation)

        # - Evaluate passthrough input layer
        res_inputs = np.dot(inputs, w_in)

        # - Compute random numbers for reservoir noise
        _, subkey = rand.split(key)
        noise = noise_std * rand.normal(subkey, shape=(inputs.shape[0], np.size(x0)))

        # - Use `scan` to evaluate reservoir
        x, (rec_inputs, res_acts) = scan(reservoir_step, x0, (res_inputs, noise))

        # - Evaluate passthrough output layer
        outputs = np.dot(res_acts, w_out)

        return x, res_inputs, rec_inputs, res_acts, outputs

    return rec_evolve_jit

def _get_rec_evolve_directly_jit(H: Callable[[float], float]):
    @jit
    def rec_evolve_jit(
        x0: np.ndarray,
        w_recurrent: np.ndarray,
        w_out: np.ndarray,
        bias: np.ndarray,
        tau: np.ndarray,
        inputs: np.ndarray,
        noise_std: float,
        key,
        dt: float,
    ):
        # - Pre-compute dt/tau
        dt_tau = dt / tau

        # - Reservoir state step function (forward Euler solver)
        def reservoir_step(x, inps):
            inp, rand = inps
            activation = H(x)
            rec_input = np.dot(activation, w_recurrent)
            dx = dt_tau * (-x + inp + bias + rand + rec_input)

            return x + dx, (rec_input, activation)

        # - Compute random numbers for reservoir noise
        __all__, subkey = rand.split(key)
        noise = noise_std * rand.normal(subkey, shape=(inputs.shape[0], np.size(x0)))

        # - Use `scan` to evaluate reservoir
        x, (rec_inputs, res_acts) = scan(reservoir_step, x0, (inputs, noise))

        # - Evaluate passthrough output layer
        outputs = np.dot(res_acts, w_out)

        return x, inputs, rec_inputs, res_acts, outputs

    return rec_evolve_jit


def _get_force_evolve_jit(H: Callable):
    @jit
    def force_evolve_jit(
        x0: np.ndarray,
        w_in: FloatVector,
        w_out: FloatVector,
        bias: FloatVector,
        tau: FloatVector,
        inputs: np.ndarray,
        force: np.ndarray,
        noise_std: float,
        key,
        dt: float,
    ):
        """
        force_evolve_jit() - Compiled recurrent evolution function

        :param x0:          np.ndarray Initial state of forced layer [N]
        :param w_in:        np.ndarray Input weights [IxN]
        :param w_recurrent: np.ndarray Recurrent weights [NxN]
        :param w_out:       np.ndarray Output weights [NxO]
        :param bias:        np.ndarray Bias values of reservoir units [N]
        :param tau:         np.ndarray Time constants of reservoir units [N]
        :param inputs:      np.ndarray Input time series [TxN]
        :param force:       np.ndarray Driving time series injected into reservoir units instead of recurrent activity [TxN]
        :param noise_std:   float Standard deviation of noise injected into reservoir units
        :param key:         Jax RNG key to use in noise generation
        :param dt:          float Time step for forward Euler solver

        :return:    (x, res_inputs, rec_inputs, res_acts, outputs)
                x:          np.ndarray State of
                res_inputs: np.ndarray Time series of weighted external inputs to each reservoir unit [TxN]
                res_acts:   np.ndarray Time series of reservoir unit activities [TxN]
                outputs:    np.ndarray Time series of output layer values [TxO]
        """
        # - Pre-compute dt/tau
        dt_tau = dt / tau

        # - Reservoir state step function (forward Euler solver)
        def reservoir_step(x, inps):
            inp, rand, force = inps
            activation = H(x)
            dx = dt_tau * (-x + inp + bias + rand + force)

            return x + dx, activation

        # - Evaluate passthrough input layer
        res_inputs = np.dot(inputs, w_in)

        # - Compute random numbers for reservoir noise
        __all__, subkey = rand.split(key)
        noise = noise_std * rand.normal(subkey, shape=(inputs.shape[0], np.size(x0)))

        # - Use `scan` to evaluate reservoir
        x, res_acts = scan(reservoir_step, x0, (res_inputs, noise, force))

        # - Evaluate passthrough output layer
        outputs = np.dot(res_acts, w_out)

        return x, res_inputs, res_acts, outputs

    return force_evolve_jit


# -- Recurrent reservoir


class RecRateEulerJax(Layer):
    """
    ``JAX``-backed firing-rate recurrent layer

    `.RecRateEulerJax` implements a recurrent reservoir with input and output weighting, using a ``JAX``-implemented solver as a back end. The design permits gradient-based learning of weights, biases and time constants using `jax.grad`.

    `.RecRateEulerJax` is compatible with the `.layers.training.train_jax_rate_sgd` module.

    .. rubric:: Dynamics

    The layer implements the dynamics

    .. math:: \\tau \\cdot \\dot{x} + x = I(t) + W_{rec} \\cdot H(x) + b + \\sigma \\cdot \\zeta(t)

    where :math:`\\tau`` is the neuron time constants; :math:`x` is the N-dimensional state of the neurons; :math:`I(t)` is the N-dimensional input signal; :math:`W_{rec}` is the NxN-dimensional matrix of recurrent synaptic connections; :math:`b` is the N-dimensional vector of bias currents for the neurons; and :math:`\\sigma \\cdot \\zeta(t)` is a white noise process with std. dev :math:`\\sigma`.

    The outputs of the layer are given by :math:`H(x)`, where :math:`H(x)` is a neuron transfer function. By default, :math:`H(x)` is the linear-threshold function

    .. math:: H_{ReLU}(x) = \\text{max}(x, 0)
    """

    def __init__(
        self,
        weights: np.ndarray,
        tau: np.ndarray,
        bias: np.ndarray,
        noise_std: float = 0.0,
        activation_func: Union[str, Callable[[FloatVector], FloatVector]] = H_ReLU,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[int] = None,
    ):
        """
        ``JAX``-backed firing rate reservoir, with input and output weighting

        :param np.ndarray weights:                  Recurrent weights [NxN]
        :param np.ndarray tau:                      Time constants [N]
        :param np.ndarray bias:                     Bias values [N]
        :param float noise_std:                     White noise standard deviation applied to reservoir neurons. Default: ``0.0``
        :param Union[str, Callable[[FloatVector], float]] activation_func:   Neuron transfer function f(x: float) -> float. Must be vectorised. Default: H_ReLU. Can be specified as a string: ['relu', 'tanh']
        :param Optional[float] dt:                  Reservoir time step. Default: ``np.min(tau) / 10.0``
        :param Optional[str] name:                  Name of the layer. Default: ``None``
        :param Optional[Jax RNG key] rng_key        Jax RNG key to use for noise. Default: Internally generated
        """

        # - Everything should be 2D
        weights = np.atleast_2d(weights)

        # transform to np.array if necessary
        tau = np.array(tau)
        bias = np.array(bias)

        # - Get information about network size
        self._size_in = weights.shape[0]
        self._size = weights.shape[1]
        self._size_out = weights.shape[1]

        # -- Set properties
        self.w_recurrent = weights
        self.tau = tau
        self.bias = bias
        self.H = activation_func

        if dt is None:
            dt = np.min(tau) / 10.0

        # - Call super-class initialisation
        super().__init__(weights, dt, noise_std, name)

        # - Get compiled evolution function
        self._evolve_jit = _get_rec_evolve_jit(self._H)
        self._evolve_directly_jit = _get_rec_evolve_directly_jit(self._H)

        # - Reset layer state
        self.reset_all()

        # - Set unit internal input and output weights, for compatibility with later layers
        if not hasattr(self, '_w_in'):
            self._w_in = 1.
        if not hasattr(self, '_w_out'):
            self._w_out = 1.

        # - Seed RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
        _, self._rng_key = rand.split(np.array(rng_key, dtype=np.uint32))

    def randomize_state(self):
        """
        Randomize the internal state of the layer.
        """
        _, subkey = rand.split(self._rng_key)
        self._state = rand.normal(subkey, shape=(self._size, ))

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> TimeSeries:
        """
        Evolve the reservoir state

        :param ts_input:        TSContinuous Input time series
        :param duration:        float Duration of evolution in seconds
        :param num_timesteps:   int Number of time steps to evolve (based on self.dt)

        :return: ts_output:     TSContinuous Output time series
        """

        # - Prepare time base and inputs
        time_base, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Call raw evolution function
        res_inputs, rec_inputs, res_acts, outputs = self._evolve_raw(inps)

        # - Store evolution time series
        self.res_inputs_last_evolution = TSContinuous(time_base, res_inputs)
        self.rec_inputs_last_evolution = TSContinuous(time_base, rec_inputs)
        self.res_acts_last_evolution = TSContinuous(time_base, res_acts)

        # - Wrap outputs as time series
        return TSContinuous(time_base, onp.array(outputs))

    def evolve_directly(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> TimeSeries:
        """
        evolve() - Evolve the reservoir state

        :param ts_input:        TSContinuous Input time series
        :param duration:        float Duration of evolution in seconds
        :param num_timesteps:   int Number of time steps to evolve (based on self.dt)

        :return: ts_output:     TSContinuous Output time series
        """
        
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)
        # - Generate discrete time base
        time_base = onp.array(self._gen_time_trace(self.t, num_timesteps))
        inps = ts_input.samples

        # - Call raw evolution function
        res_inputs, rec_inputs, res_acts, outputs = self._evolve_directly_raw(inps)

        # - Store evolution time series
        self.res_inputs_last_evolution = TSContinuous(time_base, res_inputs)
        self.rec_inputs_last_evolution = TSContinuous(time_base, rec_inputs)
        self.res_acts_last_evolution = TSContinuous(time_base, res_acts)

        # - Wrap outputs as time series
        return TSContinuous(time_base, onp.array(outputs))

    def _evolve_raw(
        self, inps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        _evolve_raw() - Raw evolution of an input array

        :param inps:    np.ndarray Input matrix [T, I]

        :return:  (res_inputs, rec_inputs, res_acts, outputs)
                res_inputs:     np.ndarray Weighted inputs to reservoir units [T, N]
                rec_inputs      np.ndarray Recurrent inputs to reservoir units [T, N]
                res_acts        np.ndarray Reservoir activity trace [T, N]
                outputs         np.ndarray Output of network [T, O]
        """
        # - Call compiled Euler solver to evolve reservoir
        self._state, res_inputs, rec_inputs, res_acts, outputs = self._evolve_jit(
            self._state,
            self._w_in,
            self._weights,
            self._w_out,
            self._bias,
            self._tau,
            inps,
            self._noise_std,
            self._rng_key,
            self._dt,
        )

        # - Increment timesteps
        self._timestep += inps.shape[0] - 1

        return res_inputs, rec_inputs, res_acts, outputs

    def _evolve_directly_raw(self, inps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._state, res_inputs, rec_inputs, res_acts, outputs = self._evolve_directly_jit(
            self._state,
            self._w_recurrent,
            self._w_out,
            self._bias,
            self._tau,
            inps,
            self._noise_std,
            self._rng_key,
            self._dt,
        )

        self._timestep += inps.shape[0] - 1

        return res_inputs, rec_inputs, res_acts, outputs

    def _prepare_input(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> (np.ndarray, np.ndarray, float):
        """
        _prepare_input - Sample input, set up time base

        :param Optional[TSContinuous] ts_input: TxM or Tx1 Input signals for this layer
        :param Optional[float] duration:        Duration of the desired evolution, in seconds
        :param Optional[int] num_timesteps:     Number of evolution time steps

        :return: (time_base, input_steps, duration)
            time_base:          ndarray T1 Discretised time base for evolution
            input_steps:        ndarray (T1xN) Discretised input signal for layer
            num_timesteps:      int Actual number of evolution time steps
        """

        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Generate discrete time base
        time_base = onp.array(self._gen_time_trace(self.t, num_timesteps))

        if ts_input is not None:
            if not ts_input.periodic:
                # - If time base limits are very slightly beyond ts_input.t_start and ts_input.t_stop, match them
                if (
                    ts_input.t_start - 1e-3 * self.dt
                    <= time_base[0]
                    <= ts_input.t_start
                ):
                    time_base[0] = ts_input.t_start
                if ts_input.t_stop <= time_base[-1] <= ts_input.t_stop + 1e-3 * self.dt:
                    time_base[-1] = ts_input.t_stop

            # - Warn if evolution period is not fully contained in ts_input
            if not (ts_input.contains(time_base) or ts_input.periodic):
                warn(
                    "Layer `{}`: Evolution period (t = {} to {}) ".format(
                        self.name, time_base[0], time_base[-1]
                    )
                    + "not fully contained in input signal (t = {} to {})".format(
                        ts_input.t_start, ts_input.t_stop
                    )
                )

            # - Sample input trace and check for correct dimensions
            input_steps = self._check_input_dims(ts_input(time_base))

            # - Treat "NaN" as zero inputs
            input_steps[onp.where(np.isnan(input_steps))] = 0

        else:
            # - Assume zero inputs
            input_steps = np.zeros((np.size(time_base), self.size_in))

        return time_base, np.array(input_steps), num_timesteps

    def to_dict(self) -> dict:
        """
        Convert the parameters of this class to a dictionary

        :return dict:
        """
        config = {}
        config["class_name"] = "RecRateEulerJax"
        config["weights"] = onp.array(self.weights).tolist()
        config["tau"] = onp.array(self.tau).tolist()
        config["bias"] = onp.array(self.bias).tolist()
        config["rng_key"] = onp.array(self._rng_key).tolist()
        config["noise_std"] = (
            self.noise_std if type(self.noise_std) is float else self.noise_std.tolist()
        )
        config["dt"] = self.dt
        config["name"] = self.name

        # - Check for a supported activation function
        assert self._H == H_ReLU or self._H == H_tanh,\
            "Only models using ReLU or tanh activation functions are saveable."

        # - Encode the activation function as a string
        if self._H == H_ReLU:
            config["activation_func"] = "relu"
        elif self._H == H_tanh:
            config["activation_func"] = "tanh"
        else:
            raise(Exception)
        return config

    @property
    def H(self):
        """ (Callable) Activation function used by the neurons in this layer """
        return self._H

    @H.setter
    def H(self, value):
        if type(value) is str:
            if value in ["relu", 'ReLU', "H_ReLU"]:
                self._H = H_ReLU
            elif value in ["tanh", "TANH", "H_tanh"]:
                self._H = H_tanh
            else:
                raise ValueError('The activation function must be one of ["relu", "tanh"]')
        else:
            # - Test the activation function
            try:
                ret = value(np.array([0.1, 0.1]))
            except:
                raise TypeError('The activation function must be a Callable[[FloatVector], FloatVector]')

            assert np.size(ret) == 2, \
                'The activation function must return an array the same size as the input'

            assert type(ret) is not tuple, \
                'The activation function must not return multiple arguments'

            # - Assign the activation function
            self._H = value

    @property
    def w_recurrent(self) -> np.ndarray:
        """ (np.ndarray) [NxN] recurrent weights """
        return onp.array(self._weights)

    @w_recurrent.setter
    def w_recurrent(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_recurrent` must be 2D"

        assert value.shape == (
            self._size,
            self._size,
        ), "`w_recurrent` must be [{:d}, {:d}]".format(self._size, self._size)

        self._weights = np.array(value).astype("float")

    @property
    def tau(self) -> np.ndarray:
        return onp.array(self._tau)

    @tau.setter
    def tau(self, value: np.ndarray):
        # - Replicate `tau` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        assert (
            np.size(value) == self._size
        ), "`tau` must have {:d} elements or be a scalar".format(self._size)

        self._tau = np.reshape(value, self._size).astype("float")

    @property
    def bias(self) -> np.ndarray:
        return onp.array(self._bias)

    @bias.setter
    def bias(self, value: np.ndarray):
        # - Replicate `bias` from a scalar value
        if np.size(value) == 1:
            value = np.repeat(value, self._size)

        assert (
            np.size(value) == self._size
        ), "`bias` must have {:d} elements or be a scalar".format(self._size)

        self._bias = np.reshape(value, self._size).astype("float")

    @property
    def dt(self) -> float:
        return onp.array(self._dt).item(0)

    @dt.setter
    def dt(self, value: float):
        # - Ensure dt is numerically stable
        tau_min = np.min(self.tau) / 10.0
        if value is None:
            value = tau_min

        assert value >= tau_min, "`tau` must be at least {:.2e}".format(tau_min)

        self._dt = np.array(value).astype("float")


class RecRateEulerJax_IO(RecRateEulerJax):
    """
    ``JAX``-backed firing-rate recurrent layer, with input and output weights

    `.RecRateEulerJax_IO` implements a recurrent reservoir with input and output weighting, using a ``JAX``-implemented solver as a back end. The design permits gradient-based learning of weights, biases and time constants using `jax.grad`.

    `.RecRateEulerJax_IO` is compatible with the `.layers.training.train_jax_rate_sgd` module.

    .. rubric:: Dynamics

    The layer implements the dynamics

    .. math:: \\tau \\cdot \\dot{x} + x = W_{in} \\dot I(t) + W_{rec} \\cdot H(x) + b + \\sigma \\cdot \\zeta(t)

    where :math:`\\tau`` is the neuron time constants; :math:`x` is the N-dimensional state of the neurons; :math:`I(t)` is the I-dimensional input vector; :math:`W_{in}` is the [IxN] matrix of input weights; :math:`W_{rec}` is the NxN-dimensional matrix of recurrent synaptic connections; :math:`b` is the N-dimensional vector of bias currents for the neurons; and :math:`\\sigma \\cdot \\zeta(t)` is a white noise process with std. dev :math:`\\sigma`.

    The outputs of the layer are given by :math:`W_{out} \\cdot H(x)`, where :math:`W_{out}` is the NxM output weight matrix, and :math:`H(x)` is a neuron transfer function. By default, :math:`H(x)` is the linear-threshold function

    .. math:: H_{ReLU}(x) = \\text{max}(x, 0)
    """

    def __init__(
        self,
        w_in: np.ndarray,
        w_recurrent: np.ndarray,
        w_out: np.ndarray,
        tau: FloatVector,
        bias: FloatVector,
        noise_std: float = 0.0,
        activation_func: Callable[[FloatVector], FloatVector] = H_ReLU,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[int] = None,
    ):
        """
        RecRateEulerJax_IO - ``JAX``-backed firing rate reservoir, with input and output weighting

        :param np.ndarray w_in:                     Input weights [IxN]
        :param np.ndarray w_recurrent:              Recurrent weights [NxN]
        :param np.ndarray w_out:                    Output weights [NxO]
        :param np.ndarray tau:                      Time constants [N]
        :param np.ndarray bias:                     Bias values [N]
        :param float noise_std:           White noise standard deviation applied to reservoir neurons. Default: ``0.0``
        :param Callable[[FloatVector], float] activation_func:   Neuron transfer function f(x: float) -> float. Must be vectorised. Default: H_ReLU
        :param Optional[float] dt:                  Reservoir time step. Default: ``np.min(tau) / 10.0``
        :param Optional[str] name:                  Name of the layer. Default: ``None``
        :param Optional[Jax RNG key] rng_key        Jax RNG key to use for noise. Default: Internally generated
        """

        # - Everything should be 2D
        w_in = np.atleast_2d(w_in)
        w_recurrent = np.atleast_2d(w_recurrent)
        w_out = np.atleast_2d(w_out)

        # transform to np.array if necessary
        tau = np.array(tau)
        bias = np.array(bias)

        # - Get information about network size
        self._size_in = w_in.shape[0]
        self._size = w_in.shape[1]
        self._size_out = w_out.shape[1]

        # -- Set properties
        self.w_in = w_in
        self.w_recurrent = w_recurrent
        self.w_out = w_out
        self.tau = tau
        self.bias = bias
        self._H = activation_func

        if dt is None:
            dt = np.min(tau) / 10.0

        # - Call super-class initialisation
        super().__init__(w_recurrent, tau, bias, noise_std, activation_func, dt, name, rng_key)

        # - Correct layer size
        self._size_in = w_in.shape[0]
        self._size_out = w_out.shape[1]

    def to_dict(self) -> dict:
        """
        Convert the parameters of this class to a dictionary

        :return dict:
        """
        # - Get base dictionary
        config = super().to_dict()
        config.pop('weights')

        # - Include class-specific aspects
        config.update({
            "class_name": "RecRateEulerJax_IO",
            "w_in": onp.array(self.w_in).tolist(),
            "w_recurrent": onp.array(self.w_recurrent).tolist(),
            "w_out": onp.array(self.w_out).tolist(),
        })

        # - Return configuration
        return config

    @property
    def w_in(self) -> np.ndarray:
        """ (np.ndarray) [IxN] input weights """
        return onp.array(self._w_in)

    @w_in.setter
    def w_in(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_in` must be 2D"

        assert value.shape == (
            self._size_in,
            self._size,
        ), "`w_in` must be [{:d}, {:d}]".format(self._size_in, self._size)

        self._w_in = np.array(value).astype("float")

    @property
    def w_out(self) -> np.ndarray:
        """ (np.ndarray) [NxO] output weights """
        return onp.array(self._w_out)

    @w_out.setter
    def w_out(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_out` must be 2D"

        assert value.shape == (
            self._size,
            self._size_out,
        ), "`w_out` must be [{:d}, {:d}]".format(self._size, self._size_out)

        self._w_out = np.array(value).astype("float")


class ForceRateEulerJax_IO(RecRateEulerJax_IO):
    """
    Implements a pseudo recurrent reservoir, for use in reservoir transfer

    In this layer, input and output weights are present, but no recurrent connectivity exists. Instead, "recurrent inputs" are injected into each layer neuron. The activations of the neurons are then compared to those of a target reservoir, and the recurrent weights can be solved for using linear regression.

    .. rubric:: Dynamics

    The layer implements the dynamics

    .. math:: \\tau \\cdot \\dot{x} + x = W_{in} \\dot I(t) + F(t) + b + \\sigma \\cdot \\zeta(t)

    where :math:`\\tau`` is the neuron time constants; :math:`x` is the N-dimensional state of the neurons; :math:`I(t)` is the I-dimensional input vector; :math:`W_{in}` is the [IxN] matrix of input weights; :math:`F(t)` is the N-dimensional vector of forcing currents --- these should be the recurrent input currents taking from another reservoir; :math:`b` is the N-dimensional vector of bias currents for the neurons; and :math:`\\sigma \\cdot \\zeta(t)` is a white noise process with std. dev :math:`\\sigma`.

    The outputs of the layer are given by :math:`W_{out} \\cdot H(x)`, where :math:`W_{out}` is the NxO output weight matrix, and :math:`H(x)` is a neuron transfer function. By default, :math:`H(x)` is the linear-threshold function

    .. math:: H_{ReLU}(x) = \\text{max}(x, 0)
    """

    def __init__(
        self,
        w_in: np.ndarray,
        w_out: np.ndarray,
        tau: FloatVector,
        bias: FloatVector,
        noise_std: float = 0.0,
        activation_func: Callable[[FloatVector], FloatVector] = H_ReLU,
        dt: Optional[float] = None,
        name: Optional[str] = None,
        rng_key: Optional[int] = None,
    ):
        """
        ``JAX``-backed firing rate reservoir, used for reservoir transfer

        :param np.ndarray w_in:                     Input weights [IxN]
        :param np.ndarray w_out:                    Output weights [NxO]
        :param np.ndarray tau:                      Time constants [N]
        :param np.ndarray bias:                     Bias values [N]
        :param Optional[float] noise_std:           White noise standard deviation applied to reservoir neurons. Default: ``0.0``
        :param Callable[[FloatVector], float] activation_func:  Neuron transfer function f(x: float) -> float. Must be vectorised. Default: ``H_ReLU``
        :param Optional[float] dt:                  Reservoir time step. Default: ``np.min(tau) / 10.0``
        :param Optional[str] name:                  Name of the layer. Default: ``None``
        :param Optional[Jax RNG key] rng_key        Jax RNG key to use for noise. Default: Internally generated
        """

        # - Everything should be 2D
        w_in = np.atleast_2d(w_in)
        w_out = np.atleast_2d(w_out)

        # - Call super-class initialisation
        super().__init__(
            w_in,
            np.zeros((w_in.shape[1], w_in.shape[1])),
            w_out,
            tau,
            bias,
            noise_std,
            activation_func,
            dt,
            name,
            rng_key,
        )

        # - Get compiled evolution function for forced reservoir
        self._evolve_jit = _get_force_evolve_jit(activation_func)

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        ts_force: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
    ) -> TimeSeries:
        """
        evolve() - Evolve the reservoir state

        :param Optional[TSContinuous] ts_input: Input time series
        :param Optional[TSContinuous] ts_force: Forced time series
        :param Optional[float] duration:        Duration of evolution in seconds
        :param Optional[int] num_timesteps:     Number of time steps to evolve (based on self.dt)

        :return: ts_output:     TSContinuous Output time series
        """

        # - Prepare time base and inputs
        time_base, inps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Extract forcing inputs
        if ts_force is None:
            forces = np.zeros((num_timesteps, self._size))
        else:
            forces = ts_force(time_base)

        # - Call raw evolution function
        _, _, outputs = self._evolve_raw(inps, forces)

        # - Wrap outputs as time series
        return TSContinuous(time_base, outputs)

    def _evolve_raw(
        self, inps: np.ndarray, forces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        _evolve_raw() - Raw evolution of an input array

        :param inps:    np.ndarray Input matrix [T, I]
        :param forces:  np.ndarray Forcing signals [T, N]

        :return:  (res_inputs, rec_inputs, res_acts, outputs)
                res_inputs:     np.ndarray Weighted inputs to forced reservoir units [T, N]
                res_acts        np.ndarray Reservoir activity trace [T, N]
                outputs         np.ndarray Output of network [T, O]
        """
        # - Call compiled Euler solver to evolve reservoir
        self._state, res_inputs, res_acts, outputs = self._evolve_jit(
            self._state,
            self._w_in,
            self._w_out,
            self._bias,
            self._tau,
            inps,
            forces,
            self._noise_std,
            self._rng_key,
            self._dt,
        )

        # - Increment timesteps
        self._timestep += inps.shape[0] - 1

        return res_inputs, res_acts, outputs

    def to_dict(self) -> dict:
        """
        Convert the layer to a dictionary for saving
        :return dict:
        """
        # - Get base dictionary
        config = super().to_dict()
        config.pop('w_recurrent')

        # - Include class-specific aspects
        config.update({
            "class_name": "ForceRateEulerJax_IO",
            "w_in": onp.array(self.w_in).tolist(),
            "w_out": onp.array(self.w_out).tolist(),
        })

        # - Return configuration
        return config

class FFRateEulerJax(RecRateEulerJax):
    """
    ``JAX``-backed firing-rate recurrent layer

    `.FFRateEulerJax` implements a feed-forward dynamical layer, using a ``JAX``-implemented solver as a back end. The design permits gradient-based learning of weights, biases and time constants using `jax.grad`.

    `.FFRateEulerJax` is compatible with the `.layers.training.train_jax_rate_sgd` module.

    .. rubric:: Dynamics

    This layer implements the dynamical system

    .. math:: \\tau \\cdot \\dot{x} + x = W \\cdot i(t) + b + \\sigma\\cdot\\zeta(t)

    where :math:`\\tau`` is the neuron time constants; :math:`x` is the N-dimensional state vector of the layer neurons;
    :math:`i(t)` is the I-dimensional input signal at time :math:`t`; :math:`W` is an [IxN] matrix defining the weight matrix of this layer; :math:`b` is a vector of bias inputs for each neuron; and :math:`\\sigma\\cdot\\zeta(t)` is a white noise process with std. dev. :math:`\\sigma``.

    The output of the layer is :math:`H(x)`, where :math:`H(x)` is a neuron transfer function. By default, :math:`H(x)` is the linear-threshold function

    .. math:: H_{ReLU}(x) = \\text{max}(x, 0)
    """

    def __init__(
            self,
            w_in: np.ndarray,
            tau: np.ndarray,
            bias: np.ndarray,
            noise_std: float = 0.0,
            activation_func: Callable[[FloatVector], FloatVector] = H_ReLU,
            dt: Optional[float] = None,
            name: Optional[str] = None,
            rng_key: Optional[int] = None,
    ):
        """
        Implement a ``JAX``-backed feed-forward dynamical neuron layer.

        :param np.ndarray w_in:                 Weights [IxN]
        :param np.ndarray tau:                  Time constants [N]
        :param np.ndarray bias:                 Bias values [N]
        :param float noise_std:                 White noise standard deviation applied to reservoir neurons. Default: ``0.0``
        :param Callable[[FloatVector], float] activation_func:   Neuron transfer function f(x: float) -> float. Must be vectorised. Default: H_ReLU
        :param Optional[float] dt:              Reservoir time step. Default: ``np.min(tau) / 10.0``
        :param Optional[str] name:              Name of the layer. Default: ``None``
        :param Optional[Jax RNG key] rng_key:   Jax RNG key to use for noise. Default: Internally generated
        """

        # - Everything should be 2D
        w_in = np.atleast_2d(w_in)

        # - Transform to np.array if necessary
        tau = np.array(tau)
        bias = np.array(bias)

        if dt is None:
            dt = np.min(tau) / 10.0

        # - Call super-class initialisation
        super().__init__(0., 0., 0., noise_std, activation_func, dt, name, rng_key)

        # - Correct layer size
        self._size_in = w_in.shape[0]
        self._size = w_in.shape[1]
        self._size_out = w_in.shape[1]

        # -- Set properties
        self.tau = tau
        self.bias = bias
        self.w_in = w_in
        self._weights = 0.
        self._w_out = 1.

        # # - Get compiled evolution function
        # self._evolve_jit = _get_rec_evolve_jit(activation_func)

        # - Reset layer state
        self.reset_all()

        # # - Seed RNG
        # if rng_key is None:
        #     rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
        # _, self._rng_key = rand.split(rng_key)

    @property
    def w_in(self) -> np.ndarray:
        """ (np.ndarray) [IxN] input weights """
        return onp.array(self._w_in)

    @w_in.setter
    def w_in(self, value: np.ndarray):
        assert np.ndim(value) == 2, "`w_in` must be 2D"

        assert value.shape == (
            self._size_in,
            self._size,
        ), "`w_in` must be [{:d}, {:d}]".format(self._size_in, self._size)

        self._w_in = np.array(value).astype("float")

    def to_dict(self) -> dict:
        """
        Convert the parameters of this class to a dictionary

        :return dict:
        """
        # - Get base dictionary
        config = super().to_dict()
        config.pop('weights')

        # - Include class-specific aspects
        config.update({
            "class_name": "FFRateEulerJax",
            "w_in": onp.array(self.w_in).tolist(),
        })

        # - Return configuration
        return config
