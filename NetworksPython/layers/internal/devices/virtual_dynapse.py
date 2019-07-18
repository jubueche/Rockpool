##########
# virtual_dynapse.py - This module defines a Layer class that simulates a DynapSE
#                      processor. Its purpose is to provide an understanding of
#                      which operations are possible with the hardware. The
#                      implemented neuron model is a simplification of the actual
#                      circuits and therefore only serves as a rough approximation.
#                      Accordingly, hyperparameters such as time constants or
#                      baseweights give an idea on the parameters that can be set
#                      but there is no direct correspondence to the hardware biases.
#                      Furthermore, when connecting neurons it is possible to
#                      achieveby large fan-ins by exploiting connection aliasing.
#                      This elaborate approach has not been accounted for in this module.
# Author: Felix Bauer, aiCTX AG, felix.bauer@ai-ctx.com
##########

### --- Imports

# Built-in modules
from warnings import warn
from typing import List, Union, Optional

# Third-party modules
import numpy as np

# NetworksPython modules
from ....timeseries import TSEvent
from ...layer import Layer, ArrayLike
from ...internal.aeif_nest import RecAEIFSpkInNest
from . import params

### --- Constants
CONNECTIONS_VALID = 0
FANIN_EXCEEDED = 1
FANOUT_EXCEEDED = 2
CONNECTION_ALIASING = 4

### --- Class definition


class VirtualDynapse(Layer):

    _num_chips = params.NUM_CHIPS
    _num_cores_chip = params.NUM_CORES_CHIP
    _num_neurons_core = params.NUM_NEURONS_CORE
    _weight_resolution = 2 ** params.BIT_RESOLUTION_WEIGHTS - 1
    _stddev_mismatch = params.STDDEV_MISMATCH
    _param_names = [
        "baseweight_e",
        "baseweight_i",
        "bias",
        "refractory",
        "tau_mem_1",
        "tau_mem_2",
        "tau_syn_exc",
        "tau_syn_inh",
        "v_thresh",
        "spike_adapt",
        "tau_adapt",
        "delta_t",
        "weights_excit",
        "weights_inhib",
    ]

    def __init__(
        self,
        dt: float = 1e-5,
        connections_ext: Optional[np.ndarray] = None,
        connections_rec: Optional[np.ndarray] = None,
        tau_mem_1: Union[float, np.ndarray] = 0.02,
        tau_mem_2: Union[float, np.ndarray] = 0.02,
        has_tau2: Union[bool, np.ndarray] = False,
        tau_syn_exc: Union[float, np.ndarray] = 0.05,
        tau_syn_inh: Union[float, np.ndarray] = 0.05,
        baseweight_e: Union[float, np.ndarray] = 0.1,
        baseweight_i: Union[float, np.ndarray] = 0.1,
        bias: Union[float, np.ndarray] = 0,
        refractory: Union[float, np.ndarray] = 0.001,
        v_thresh: Union[float, np.ndarray] = 0.01,
        spike_adapt: Union[float, np.ndarray] = 0.0,
        tau_adapt: Union[float, np.ndarray] = 0.01,
        delta_t: Union[float, np.ndarray] = 2.0,
        name: str = "unnamed",
        num_threads: int = 1,
        mismatch: bool = True,
    ):
        """
        VritualDynapse - Simulation of DynapSE neurmorphic processor.
        :param dt:        Time step size in seconds
        :param connections_ext:   2D-array defining connections from external input
                                 Size at most 1024x4096. Will be filled with 0s if smaller.
        :param connections_rec:  2D-array defining connections between neurons
                                 Size at most 4096x4096. Will be filled with 0s if smaller.
        :param tau_mem_1:        float or 1D-array of size 16, with membrane time constant
                                 for each core, in seconds. If float, same for all cores.
        :param tau_mem_2:        float or 1D-array of size 16 with alternative membrane time
                                 constant for each core, in seconds. If float, same for all
                                 cores.
        :param has_tau2:         bool or 1D-array of size 4096, indicating which neuron
                                 usees the alternative membrane time constant. If bool,
                                 same for all cores.
        :param tau_syn_exc:      float or 1D-array of size 16 with time constant for
                                 excitatory synapses for each core, in seconds. If float,
                                 same for all cores.
        :param tau_syn_inh:      float or 1D-array of size 16 with time constant for
                                 inhibitory synapses for each core, in seconds. If float,
                                 same for all cores.
        :param baseweight_e:     float or 1D-array of size 16 with multiplicator (>=0) for
                                 binary excitatory weights for each core. If float, same
                                 for all cores.
        :param baseweight_i:     float or 1D-array of size 16 with multiplicator (>=0) for
                                 binary inhibitory weights for each core. If float, same
                                 for all cores.
        :param bias:             float or 1D-array of size 16 with constant neuron bias (>=0)
                                 for each core. If float, same for all cores.
        :param refractory:       float or 1D-array of size 16 with refractory time in
                                 secondsfor each core. If float, same for all cores.
        :param v_thresh:         float or 1D-array of size 16 with neuron firing v_thresh
                                 for each core. If float, same for all cores.
        :param spike_adapt:      Scaling for spike triggered adaptation.
        :param tau_adapt:        Adaptation time constant.
        :param delta_t:          Scaling for exponential part of activation function.
        :param name:             Name for this object instance.
        :param num_threads:      Number of cpu cores available for simulation.
        :param mismatch:         If True, parameters for each neuron are drawn from Gaussian
                                 around provided values for core.
                                 If an array is passed, it must be of shape
                                 (12 + 2*num_neurons x num_neurons) and provide individual
                                 mismatch factors for each parameter and neuron as well as
                                 excitatory and inhibitory weights. Order, row-wise:
                                 baseweight_e, baseweight_i, bias, refractory, tau_mem_1,
                                 tau_mem_2, tau_syn_exc, tau_syn_inh, v_thresh,
                                 weights_excit, weights_inhib
        """
        # - Handle provided mismatch
        if isinstance(mismatch, bool):
            # - Draw values for mismatch
            self._draw_mismatch(mismatch)
        else:
            num_params_noweights = len(self._spike_adapt) - 2
            shape_mismatch = (
                num_params_noweights + 2 * self.num_neurons,
                self.num_neurons,
            )
            # - Use individual mismatch factors
            if mismatch.shape == shape_mismatch:
                # - Store mismatch in dict (except for weights)
                self._mismatch_factors = {
                    param: factors
                    for param, factors in zip(
                        self._param_names[:-2], mismatch[:num_params_noweights]
                    )
                }
                # - Mismatch for weights
                self._mismatch_factors["weights_excit"] = mismatch[
                    -2 * self.num_neurons : -self.num_neurons
                ]
                self._mismatch_factors["weights_inhib"] = mismatch[-self.num_neurons :]
            else:
                warn(
                    self.start_print
                    + "`mismatch` must be array of shape {} x {}. ".format(
                        *shape_mismatch
                    )
                    + "Will generate new mismatch."
                )
                self._draw_mismatch(mismatch)

        # - Settings wrt connection validation
        self.validate_fanin = True
        self.validate_fanout = True
        self.validate_aliasing = True

        # - Store internal parameters
        self.name = name
        self._num_threads = num_threads

        # - Set up membrane time constants
        self._tau_mem_1, self._tau_mem_1_ = self._process_parameter(
            tau_mem_1, "tau_mem_1", nonnegative=True
        )
        self._tau_mem_2, self._tau_mem_2_ = self._process_parameter(
            tau_mem_2, "tau_mem_2", nonnegative=True
        )
        has_tau2 = self._expand_to_size(has_tau2, self.num_neurons, "has_tau2", False)
        if not has_tau2.dtype == bool:
            raise ValueError(self.start_print + "`has_tau2` must consist of booleans.")
        else:
            self._has_tau2 = has_tau2

        # - Set up connections and weights
        self._baseweight_e, self._baseweight_e_syn = self._process_parameter(
            baseweight_e, "weights_excit", True
        )
        self._baseweight_i, self._baseweight_i_syn = self._process_parameter(
            baseweight_i, "weights_inhib", True
        )
        self._connections_rec = self._process_connections(
            connections=connections_rec, external=False
        )
        self._connections_ext = self._process_connections(
            connections=connections_ext, external=True
        )
        if (
            (connections_rec is not None or connections_ext is not None)
            and self.validate_connections(
                self._connections_rec,
                self._connections_ext,
                verbose=True,
                validate_fanin=self.validate_fanin,
                validate_fanout=self.validate_fanout,
                validate_aliasing=self.validate_aliasing,
            )
            != CONNECTIONS_VALID
        ):
            raise ValueError(
                self.start_print + "Connections not compatible with hardware."
            )
        weights_rec = self._generate_weights(external=False)
        weights_ext = self._generate_weights(external=True)

        # - Remaining parameters: Scale up to neuron size and add mismatch
        self._bias, bias = self._process_parameter(bias, "bias", True)
        self._v_thresh, v_thresh = self._process_parameter(v_thresh, "v_thresh", True)
        self._refractory, refractory = self._process_parameter(
            refractory, "refractory", True
        )
        self._tau_syn_exc, tau_syn_exc = self._process_parameter(
            tau_syn_exc, "tau_syn_exc", True
        )
        self._tau_syn_inh, tau_syn_inh = self._process_parameter(
            tau_syn_inh, "tau_syn_inh", True
        )
        self._spike_adapt, spike_adapt = self._process_parameter(
            spike_adapt, "spike_adapt", True
        )
        self._tau_adapt, tau_adapt = self._process_parameter(
            tau_adapt, "tau_adapt", True
        )
        self._delta_t, delta_t = self._process_parameter(delta_t, "delta_t", True)

        # - Nest-layer for approximate simulation of neuron dynamics
        self._simulator = RecAEIFSpkInNest(
            weights_in=weights_ext,
            weights_rec=weights_rec,
            bias=bias,
            v_thresh=v_thresh,
            v_reset=0,
            v_rest=0,
            tau_mem=self._tau_mem_,
            tau_syn_exc=tau_syn_exc,
            tau_syn_inh=tau_syn_inh,
            spike_adapt=spike_adapt,
            tau_adapt=tau_adapt,
            subthresh_adapt=0.0,
            delta_t=delta_t,
            dt=dt,
            name=self.name + "_nest_backend",
            num_cores=num_threads,
            record=False,
        )

    def _draw_mismatch(self, consider_mismatch: bool = True):
        """
        _draw_mismatch - For each neuron and parameter draw a mismatch factor that
                         individual neuron parameters will be multiplied with. Store
                         factors in dict.
                         Parameters are drawn from a Gaussian around 1 with standard
                         deviation = `self.stddev_mismatch` and truncated at 0.1 to
                         avoid too small values.
        :param consider_mismatch:  If `False`, mismatch factors are all 1, corresponding
                                   to not having any mismatch.
        """

        if consider_mismatch:
            # - Number of mismatch factors per neuron
            num_params_noweights = len(self._param_names) - 2
            num_factors = num_params_noweights + 2 * self.num_neurons
            # - Draw mismatch factor for each neuron
            mismatch_factors = (
                1
                + np.random.randn(num_factors, self.num_neurons) * self.stddev_mismatch
            )
            # - Make sure values are not too small
            mismatch_factors = np.clip(mismatch_factors, 0.1, None)

            # - Store mismatch in dict (except for weights)
            self._mismatch_factors = {
                param: factors
                for param, factors in zip(
                    self._param_names[:-2], mismatch_factors[:num_params_noweights]
                )
            }
            # - Mismatch for weights
            self._mismatch_factors["weights_excit"] = mismatch_factors[
                -2 * self.num_neurons : -self.num_neurons
            ]
            self._mismatch_factors["weights_inhib"] = mismatch_factors[
                -self.num_neurons :
            ]
        else:
            # - Factor 1 corresponds to no mismatch at all.
            self._mismatch_factors = {param: 1 for param in self.param_names}

    def add_mismatch(self, mean_values: np.ndarray, param_name: str):
        if param_name in ("weights_ext", "weights_rec"):
            # - Separate inhibitory and excitatory weights
            weights_excit = np.clip(mean_values, 0, None)
            weights_inhib = np.clip(mean_values, None, 0)
            # - Add mismatch
            weights_excit *= self.mismatch_factors["weights_excit"]
            weights_inhib *= self.mismatch_factors["weights_inhib"]
            # - Combine weights and return
            return weights_excit + weights_inhib
        else:
            # - Multiply mismatch factors
            return mean_values * self.mismatch_factors[param_name]

    def set_connections(
        self,
        connections: Union[dict, np.ndarray],
        neurons_pre: np.ndarray,
        neurons_post: Optional[np.ndarray] = None,
        external: bool = False,
        add: bool = False,
    ):
        """
        set_connections - Set connections between specific neuron populations.
                          Verify that connections are supported by the hardware.
        :param connections:    2D np.ndarray: Will assume positive (negative) values
                               correspond to excitatory (inhibitory) synapses.
                               Axis 0 (1) corresponds to pre- (post-) synaptic neurons.
                               Sizes must match `neurons_pre` and `neurons_post`.
        :param neurons_pre:    Array-like with IDs of presynaptic neurons that `connections`
                               refer to. If None, use all neurons (from 0 to
                               self.num_neurons - 1).
        :param neurons_post:   Array-like with IDs of postsynaptic neurons that `connections`
                               refer to. If None, use same IDs as presynaptic neurons, unless
                               `external` is True. In this case use all neurons.
        :param add:            If True, new connections are added to exising ones, otherweise
                               connections between given neuron populations are replaced.
        :param external:       If True, presynaptic neurons are external.
        """

        if neurons_pre is None:
            neurons_pre = np.arange(self.num_external if external else self.size)

        if neurons_post is None:
            neurons_post = neurons_pre if not external else np.arange(self.size)

        # - Complete connectivity array after adding new connections
        connections = self.connections_ext if external else self.connections_rec

        # - Indices of specified neurons in full connectivity matrix
        ids_row, ids_col = np.meshgrid(neurons_pre, neurons_post, indexing="ij")

        if add:
            # - Add new connections to connectivity array
            connections[ids_row, ids_col] += connections
        else:
            # - Replace connections between given neurons with new ones
            connections[ids_row, ids_col] = connections

    def validate_connections(
        self,
        connections_rec: np.ndarray,
        connections_ext: Optional[np.ndarray] = None,
        neurons_pre: Optional[np.ndarray] = None,
        neurons_post: Optional[np.ndarray] = None,
        channels_ext: Optional[np.ndarray] = None,
        verbose: bool = True,
        validate_fanin: bool = True,
        validate_fanout: bool = True,
        validate_aliasing: bool = True,
    ) -> int:
        """
        validate_connections - Check whether connections are compatible with the
                               following constraints:
                                 - Fan-in per neuron is limited to 64
                                 - Fan-out of each neuron can only comprise 3 chips.
                                 - If any two neurons with presynaptic connections
                                   to any of the neurons on a given core are on
                                   different chips but have the same IDs within their
                                   respective chips, aliasing of events may occur.
                                   Therefore this scenario is considered invalid.
        :param connections_rec:     2D np.ndarray: Connectivity matrix to be
                                validated. Will assume positive (negative) values
                                correspond to excitatory (inhibitory) synapses.
        :param connections_ext:  If not `None`, 2D np.ndarray that is considered
                                as external input connections to the population
                                that `connections_rec` refers to. This is considered
                                for validaiton of the fan-in of `connections_rec`.
                                Positive (negative) values correspond to excitatory
                                (inhibitory) synapses.
        :param neurons_pre:     IDs of presynaptic neurons. If `None` (default), IDs
                                are assumed to be 0,..,`connections_rec.shape[0]`.
                                If not `None`, connections to neurons that are not
                                included in `neurons_pre` are assumed to be 0.
        :param neurons_post:    IDs of postsynaptic neurons. If `None` (default), IDs
                                are assumed to be 0,..,`connections_rec.shape[1]`.
                                If not `None`, connections from neurons that are not
                                included in `neurons_post` are assumed to be 0.
        :param channels_ext:     IDs of external input channels. If `None` (default), IDs
                                are assumed to be 0,..,`connections_ext.shape[0]`.
                                If not `None`, connections from channels that are not
                                included in `channels_ext` are assumed to be 0.
        :param verbose:         If `True`, print out detailed information about
                                validity of connections.
        :param validate_fanin:     If `True`, test if connections have valid fan-in.
        :param validate_fanout:    If `True`, test if connections have valid fan-out.
        :param validate_aliasing:  If `True`, test for connection aliasing.
        return
            Integer indicating the result of the validation:
            If displayed as a binary number, each digit corresponds to the result
            of one test (order from small to high base: fan-in, fan-out, aliasing),
            with 0 meaning passed.
            E.g. 6 (110) means that the fan-in is valid but not the fan-out and
            there is connection aliasing.
        """
        connections_rec = np.asarray(connections_rec)
        if connections_rec.ndim != 2:
            raise ValueError(
                self.start_print + "`connections_rec` must be 2-dimensional."
            )

        # - Use neuron IDs to expand on-chip connectivity matrix
        if neurons_pre is not None or neurons_post is not None:
            if neurons_pre is None:
                neurons_pre = np.arange(connections_rec.shape[0])
            if neurons_post is None:
                neurons_post = np.arange(connections_rec.shape[1])
            # - Make sure, dimensions of connection matrix and neuron arrays match
            if (
                np.size(neurons_pre) != connections_rec.shape[0]
                or np.size(neurons_post) != connections_rec.shape[1]
            ):
                raise ValueError(
                    self.start_print
                    + "Dimensions of `connections_rec` must match sizes "
                    + "of `neuron_ids_pre` and `neuron_ids_post."
                )
            # - Expand on-chip connectivity matrix to represent chip and core dimensions
            # Number of rows is large enough so that when appending external connections
            # they are recognized as not being on the same chip (for aliasing test)
            num_rows = int(
                np.ceil((np.amax(neurons_pre) + 1) / self.num_neurons_chip)
                * self.num_neurons_chip
            )
            connections_rec_0 = np.zeros((num_rows, np.amax(neurons_post) + 1))
            idcs_pre, idcs_post = np.meshgrid(neurons_pre, neurons_post, indexing="ij")
            connections_rec_0[idcs_pre, idcs_post] = connections_rec
            connections_rec = connections_rec_0

        # - Get 2D matrix with number of connections between neurons
        conn_count_onchip = self.get_connection_counts(connections_rec)

        # - Handle external connections
        if connections_ext is not None:
            connections_ext = np.asarray(connections_ext)
            if connections_ext.ndim != 2:
                raise ValueError(
                    self.start_print + "`connections_ext` must be 2-dimensional."
                )

            # - Expand external connectivity matrix to match channel IDs
            if channels_ext is not None:
                if np.size(channels_ext) != connections_ext.shape[0]:
                    raise ValueError(
                        self.start_print
                        + "Dimensions of `connections_ext` must match size of `neuron_ids_ext`."
                    )
                if neurons_post is None:
                    neurons_post = np.arange(connections_rec.shape[1])
                connections_ext_0 = np.zeros(
                    (np.amax(channels_ext) + 1, connections_rec.shape[1])
                )
                idcs_pre, idcs_post = np.meshgrid(
                    channels_ext, neurons_post, indexing="ij"
                )
                connections_ext_0[idcs_pre, idcs_post] = connections_ext
                connections_ext = connections_ext_0
            else:
                # - Verify that connection matrices match in size
                if connections_ext.shape[1] != connections_rec.shape[0]:
                    raise ValueError(
                        self.start_print
                        + "2nd axis of `connections_ext`"
                        + " and 1st axis of `connections_rec` must match."
                    )
            if connections_ext.shape[0] > self.num_neurons_chip:
                raise ValueError(
                    self.start_print
                    + f"There can be at most {self.num_neurons_chip} external channels. "
                )
            conn_count_ext = self.get_connection_counts(connections_ext)
            conn_count_full = np.vstack((conn_count_onchip, conn_count_ext))
        else:
            conn_count_full = conn_count_onchip

        # - Initialize Result to 0 (meaning all good)
        result = 0

        if verbose:
            print(self.start_print + "Testing provided connections:")

        if validate_fanin:
            # - Test fan-in
            exceeds_fanin: np.ndarray = (
                np.sum(conn_count_full, axis=0) > params.NUM_CAMS_NEURON
            )
            if exceeds_fanin.any():
                result += FANIN_EXCEEDED
                if verbose:
                    print(
                        "\tFan-in ({}) exceeded for neurons: {}".format(
                            params.NUM_CAMS_NEURON, np.where(exceeds_fanin)[0]
                        )
                    )

        if validate_fanout:
            # - Test fan-out
            # List with target chips for each (presynaptic) neuron
            tgtchip_list = [
                np.nonzero(row)[0] // self.num_neurons_chip for row in conn_count_onchip
            ]
            # Number of different target chips per neuron
            nums_tgtchips = np.array(
                [np.unique(tgtchips).size for tgtchips in tgtchip_list]
            )
            exceeds_fanout = nums_tgtchips > params.NUM_SRAMS_NEURON
            if exceeds_fanout.any():
                result += FANOUT_EXCEEDED
                if verbose:
                    print(
                        "\tEach neuron can only have postsynaptic connections to "
                        "{} chips. This limit is exceeded for neurons: {}".format(
                            params.NUM_SRAMS_NEURON, np.where(exceeds_fanout)[0]
                        )
                    )

        if validate_aliasing:
            # - Test for connection aliasing
            # Lists for collecting affected presyn. neurons (chips, IDs) and postsyn. cores
            alias_pre_chips: List[List[np.ndarray]] = []
            alias_pre_ids: List[List[int]] = []
            alias_post_cores: List[int] = []
            # - Iterate over postsynaptic cores
            for core_id in range(self.num_threads):
                # - IDs of neurons where core starts and ends
                id_start = core_id * self.num_neurons_core
                id_end = (core_id + 1) * self.num_neurons_core
                # - Connections with postsynaptic connections to this core
                conns_to_core = conn_count_full[:, id_start:id_end]
                # - Lists for collecting affected chip and neuron IDs for this core
                alias_pre_ids_core = []
                alias_pre_chips_core = []
                # - Iterate over presynaptic neuron IDs (wrt chip)
                for neuron_id in range(self.num_neurons_chip):
                    # - Connections with `neuron_id`-th neuron of each chip as presyn. neuron
                    conns_tocore_this_id = conns_to_core[
                        neuron_id :: self.num_neurons_chip
                    ]
                    # - IDs of chips from which presynaptic connecitons originate
                    connected_presyn_chips = np.unique(
                        np.nonzero(conns_tocore_this_id)[0]
                    )
                    # - Only one presynaptic chip is allowed for each core and `neuron_id`
                    #   If there are more, collect information
                    if len(connected_presyn_chips) > 1:
                        alias_pre_ids_core.append(neuron_id)
                        alias_pre_chips_core.append(connected_presyn_chips)
                if alias_pre_ids_core:
                    alias_post_cores.append(core_id)
                    alias_pre_ids.append(alias_pre_ids_core)
                    alias_pre_chips.append(alias_pre_chips_core)

            if alias_pre_chips:
                result += CONNECTION_ALIASING
                if verbose:
                    print_output = (
                        "\tConnection aliasing detected: Neurons on the same core should not "
                        + "have presynaptic connections with neurons that have same IDs (within "
                        + "their respective chips) but are on different chips. Affected "
                        + "postsynaptic cores are: "
                    )
                    for core_id, neur_ids_core, chips_core in zip(
                        alias_post_cores, alias_pre_ids, alias_pre_chips
                    ):
                        core_print = f"\tCore {core_id}:"
                        for id_neur, chips in zip(neur_ids_core, chips_core):
                            id_print = "\t\t Presynaptic ID {} on chips {}".format(
                                id_neur, ", ".join(str(id_ch) for id_ch in chips)
                            )
                            core_print += "\n" + id_print
                        print_output += "\n" + core_print
                    print(print_output)
                    if connections_ext is not None and any(
                        any(self.num_chips in chip_arr for chip_arr in sublist)
                        for sublist in alias_pre_chips
                    ):
                        print(f"\t(Chip ID {self.num_chips} refers to external input.)")

        if verbose and result == CONNECTIONS_VALID:
            print("\tConnections ok.")

        return result

    def get_connection_counts(self, connections: np.ndarray) -> np.ndarray:
        """
        get_connection_counts - From an array of connections generate a 2D matrix
                                with the total number of connections between each
                                neuron pair.
        :params connections:  2D np.ndarray (NxN): Will assume positive (negative)
                              values correspond to excitatory (inhibitory) synapses.
        :return:
            2D np.ndarray with total number of connections between neurons
        """
        # - Split into excitatory and inhibitory connections
        excit_conns = np.clip(connections, 0, None)
        inhib_conns = np.abs(np.clip(connections, None, 0))
        # - Divide by max. weight to count number of actual connections
        count_excit_conns = np.ceil(excit_conns / self.weight_resolution).astype(int)
        count_inhib_conns = np.ceil(inhib_conns / self.weight_resolution).astype(int)
        return count_excit_conns + count_inhib_conns

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param ts_input:       TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps   int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:               TSEvent  output spike series

        """
        return self._simulator.evolve(ts_input, duration, num_timesteps, verbose)

    def reset_time(self):
        self._simulator.reset_time()

    def reset_state(self):
        self._simulator.reset_state()

    def reset_all(self):
        self._simulator.reset_all()

    def _generate_weights(self, external: bool = False) -> np.ndarray:
        """
        _generate_weights: Generate weight matrix from connections and base weights
        :param external: If `True`, generate input weights, otherwise internal weights
        :return:
            2D-array with generated weights
        """
        # - Choose between input and internal connections
        connections = self.connections_ext if external else self.connections_rec
        # - Separate excitatory and inhibitory connections
        connections_excit = np.clip(connections, 0, None)
        connections_inhib = np.clip(connections, None, 0)
        # - Calculate weights
        weights_excit = connections_excit * self._baseweight_e_syn
        weights_inhib = connections_inhib * self._baseweight_i_syn
        return weights_excit + weights_inhib

    def _update_weights(self, external: bool = False, recurrent: bool = True):
        """
        _update_weights - Update internal representation of weights by multiplying
                          baseweights with number of connections for each core and
                          synapse type and add mismatch.
        :param external:  If `True`, update input weights
        :param external:  If `True`, update internal (recurrent) weights
        """
        if external:
            # - Generate external weights from connections and base weights
            weights_ext = self._generate_weights(external=True)
            # - Mismatch and weight update
            self._simulator.weights_in = weights_ext
        if recurrent:
            # - Generate recurrent weights from connections and base weights
            weights_rec = self._generate_weights(external=False)
            # - Weight update
            self._simulator.weights_rec = weights_rec

    def _process_parameter(
        self,
        parameter: Union[bool, int, float, ArrayLike],
        name: str,
        nonnegative: bool = True,
    ) -> (np.ndarray, np.ndarray):
        """
        _process_parameter - Reshape parameter to array of size `self.num_threads`.
                             If `nonnegative` is `True`, clip negative values to 0.
        :param parameter:    Parameter to be reshaped and possibly clipped.
        :param name:         Name of the paramter (for print statements).
        :param nonnegative:  If `True`, clip negative values to 0 and warn if
                             there are any.
        :return:
            core_params:  ndarray of size `self.num_threads` with parameters for each core
            neruon_params: ndarray of size `self.num_neurons` with parameters for each
                           neuron and added mismatch
        """
        core_params = self._expand_to_size(parameter, self.num_threads, name, False)
        if nonnegative:
            # - Make sure that parameters are nonnegative
            if (np.array(core_params) < 0).any():
                warn(
                    self.start_print
                    + f"`{name}` must be at least 0. Negative values are clipped to 0."
                )
            core_params = np.clip(core_params, 0, None)
        # - Expand to number of neurons
        neuron_params = np.repeat(core_params, self.num_neurons_core)
        if name in ("weights_excit", "weights_inhib"):
            # - Expand to number of (potential) synapses
            syn_params = np.repeat(
                neuron_params.reshape(1, -1), self.num_neurons, axis=0
            )
            # - Different mismatch for each synapse
            syn_params = self.add_mismatch(syn_params, name)
        else:
            # - Add mismatch for each neuron
            neuron_params = self.add_mismatch(neuron_params, name)

        return core_params, neuron_params

    def _process_connections(
        self, connections: Optional[np.ndarray] = None, external: bool = False
    ) -> np.ndarray:
        """
        _process_connections - Bring connectivity matric into correct shape
        :param connections:  Connectivity matrix. If None, generate 0-matrix.
        :external:           If True, generate external input connection matrix.
        :return:
            2D-np.ndarray - Generated connection matrix.
        """
        num_rows = self.num_external if external else self.num_neurons
        if connections is None:
            # - Remove all connections
            return np.zeros((num_rows, self.num_neurons))
        else:
            # - Handle smaller connectivity matrices by filling up with zeros
            conn_shape = connections.shape
            if conn_shape != (num_rows, self.num_neurons):
                connections0 = np.zeros((num_rows, self.num_neurons))
                connections0[: conn_shape[0], : conn_shape[1]] = connections
                connections = connections0
            return connections

    def to_dict(self) -> dict:
        """
        to_dict - Convert parameters of `self` to a dict if they are relevant for
                  reconstructing an identical layer.
        """
        config = {}
        config["dt"] = self.dt
        config["connections_ext"] = self.connections_ext.tolist()
        config["connections_rec"] = self.connections_rec.tolist()
        config["tau_mem_1"] = self.tau_mem_1.tolist()
        config["tau_mem_2"] = self.tau_mem_2.tolist()
        config["has_tau2"] = self.has_tau2.tolist()
        config["tau_syn_exc"] = self.tau_syn_exc.tolist()
        config["tau_syn_exc"] = self.tau_syn_exc.tolist()
        config["baseweight_e"] = self.baseweight_e.tolist()
        config["baseweight_i"] = self.baseweight_i.tolist()
        config["bias"] = self.bias.tolist()
        config["refractory"] = self.refractory.tolist()
        config["v_thresh"] = self.v_thresh.tolist()
        config["name"] = self.name
        config["num_threads"] = self.num_threads
        config["mismatch"] = self._mismatch_factors.tolist()

    @property
    def connections_rec(self):
        return self._connections_rec

    @connections_rec.setter
    def connections_rec(self, connections_new: Optional[np.ndarray]):
        # - Bring connections into correct shape
        connections_new = self._process_connections(connections_new, external=True)
        # - Make sure that connections have match hardware specifications
        if (
            self.validate_connections(
                connections_new,
                self.connections_ext,
                verbose=True,
                validate_fanin=self.validate_fanin,
                validate_fanout=self.validate_fanout,
                validate_aliasing=self.validate_aliasing,
            )
            == CONNECTIONS_VALID
        ):
            # - Update connections
            self._connections_rec = connections_new
            # - Update weights accordingly
            self._update_weights()
        else:
            raise ValueError(
                self.start_print + "Connections not compatible with hardware."
            )

    @property
    def connections_ext(self):
        return self._connections_ext

    @connections_ext.setter
    def connections_ext(self, connections_new: Optional[np.ndarray]):
        # - Bring connections into correct shape
        connections_new = self._process_connections(connections_new, external=True)
        # - Make sure that connections have match hardware specifications
        if (
            self.validate_connections(
                self.connections_rec,
                connections_new,
                verbose=True,
                validate_fanin=self.validate_fanin,
                validate_fanout=self.validate_fanout,
                validate_aliasing=self.validate_aliasing,
            )
            == CONNECTIONS_VALID
        ):
            raise ValueError(
                self.start_print + "Connections not compatible with hardware."
            )
        else:
            # - Update connections
            self._connections_ext = connections_new
            # - Update weights accordingly
            self._update_weights(external=True)

    @property
    def weights(self):
        return self._simulator.weights

    @property
    def weights_rec(self):
        return self._simulator.weights_rec

    @property
    def weights_ext(self):
        return self._simulator.weights_in

    @property
    def baseweight_e(self):
        return self._baseweight_e

    @baseweight_e.setter
    def baseweight_e(self, baseweight_new: Union[float, ArrayLike]):
        self._baseweight_e, self._baseweight_e_syn = self._process_parameter(
            baseweight_new, "weights_excit", nonnegative=True
        )
        self._update_weights(external=True, recurrent=True)

    @property
    def baseweight_i(self):
        return self._baseweight_i

    @baseweight_i.setter
    def baseweight_i(self, baseweight_new: Union[float, ArrayLike]):
        self._baseweight_i, self._baseweight_i_syn = self._process_parameter(
            baseweight_new, "weights_inhib", nonnegative=True
        )
        self._update_weights(external=True, recurrent=True)

    @property
    def bias(self):
        return self._bias

    @property
    def bias_(self):
        return self._simulator.bias

    @bias.setter
    def bias(self, bias_new: Union[float, ArrayLike]):
        self._bias, self._simulator.bias = self._process_parameter(
            bias_new, "bias", nonnegative=True
        )

    @property
    def refractory(self):
        return self._refractory

    @property
    def refractory_(self):
        return self._simulator.refractory

    @refractory.setter
    def refractory(self, t_refractory_new: Union[float, ArrayLike]):
        self._refractory, self._simulator.refractory = self._process_parameter(
            t_refractory_new, "refractory", nonnegative=True
        )

    @property
    def v_thresh(self):
        return self._v_thresh

    @property
    def v_thresh_(self):
        return self._simulator.v_thresh

    @v_thresh.setter
    def v_thresh(self, threshold_new: Union[float, ArrayLike]):
        self._v_thresh, self._simulator.v_thresh = self._process_parameter(
            threshold_new, "v_thresh", nonnegative=True
        )

    @property
    def delta_t(self):
        return self._delta_t

    @property
    def delta_t_(self):
        return self._simulator.delta_t

    @delta_t.setter
    def delta_t(self, new_delta: Union[float, ArrayLike]):
        self._delta_t, self._simulator.delta_t = self._process_parameter(
            new_delta, "delta_t", nonnegative=True
        )

    @property
    def spike_adapt(self):
        return self._spike_adapt

    @property
    def spike_adapt_(self):
        return self._simulator.spike_adapt

    @spike_adapt.setter
    def spike_adapt(self, new_const: Union[float, ArrayLike]):
        self._spike_adapt, self._simulator.spike_adapt = self._process_parameter(
            new_const, "spike_adapt", nonnegative=True
        )

    @property
    def tau_adapt(self):
        return self._tau_adapt

    @property
    def tau_adapt_(self):
        return self._simulator.tau_adapt

    @tau_adapt.setter
    def tau_adapt(self, new_tau: Union[float, ArrayLike]):
        self._tau_adapt, self._simulator.tau_adapt = self._process_parameter(
            new_tau, "tau_adapt", nonnegative=True
        )

    @property
    def tau_mem_1(self):
        return self._tau_mem_1

    @tau_mem_1.setter
    def tau_mem_1(self, tau_mem_1_new: Union[float, ArrayLike]):
        self._tau_mem_1, self._tau_mem_1_ = self._process_parameter(
            tau_mem_1_new, "tau_mem_1", nonnegative=True
        )
        # - Update simulator
        self._simulator.tau_mem = self._tau_mem_

    @property
    def tau_mem_2(self):
        return self._tau_mem_2

    @tau_mem_2.setter
    def tau_mem_2(self, tau_mem_2_new: Union[float, ArrayLike]):
        self._tau_mem_2, self._tau_mem_2_ = self._process_parameter(
            tau_mem_2_new, "tau_mem_2", nonnegative=True
        )
        # - Update simulator
        self._simulator.tau_mem = self._tau_mem_

    @property
    def has_tau2(self):
        return self._has_tau2

    @has_tau2.setter
    def has_tau2(self, new_has_tau2: Union[bool, ArrayLike]):
        new_has_tau2 = self._expand_to_size(
            new_has_tau2, self.num_neurons, "has_tau2", False
        )
        if not new_has_tau2.dtype == bool:
            raise ValueError(self.start_print + "`has_tau2` must consist of booleans.")
        else:
            self._has_tau2 = new_has_tau2
        # - Update simulator
        self._simulator.tau_mem = self._tau_mem_

    @property
    def _tau_mem_(self):
        # - Drop neurons with other time constant assigned
        tau_mem_1_ = self._tau_mem_1_ * (self.has_tau2 == False)
        tau_mem_2_ = self._tau_mem_2_ * self.has_tau2
        # - Join time constants
        return tau_mem_1_ + tau_mem_2_

    @property
    def tau_mem_(self):
        return self._simulator.tau_mem

    @property
    def tau_syn_exc(self):
        return self._tau_syn_exc

    @property
    def tau_syn_exc_(self):
        return self._simulator.tau_syn_exc

    @tau_syn_exc.setter
    def tau_syn_exc(self, tau_syn_e_new: Union[float, ArrayLike]):
        self._tau_syn_exc, self._simulator.tau_syn_exc = self._process_parameter(
            tau_syn_e_new, "tau_syn_exc", nonnegative=True
        )

    @property
    def tau_syn_inh(self):
        return self._tau_syn_inh

    @property
    def tau_syn_inh_(self):
        return self._simulator.tau_syn_inh

    @tau_syn_inh.setter
    def tau_syn_inh(self, tau_syn_i_new: Union[float, ArrayLike]):
        self._tau_syn_inh, self._simulator.tau_syn_inh = self._process_parameter(
            tau_syn_i_new, "tau_syn_inh", nonnegative=True
        )

    @property
    def num_chips(self):
        return self._num_chips

    @property
    def num_cores_chip(self):
        return self._num_cores_chip

    @property
    def num_neurons_core(self):
        return self._num_neurons_core

    @property
    def num_threads(self):
        return self._num_cores_chip * self._num_chips

    @property
    def num_neurons_chip(self):
        return self.num_cores_chip * self._num_neurons_core

    @property
    def num_neurons(self):
        return self.num_neurons_chip * self._num_chips

    @property
    def num_external(self):
        return self.num_neurons_chip

    @property
    def weight_resolution(self):
        return self._weight_resolution

    @property
    def start_print(self):
        return f"VirtualDynapse '{self.name}': "

    @property
    def input_type(self):
        return TSEvent

    @property
    def output_type(self):
        return TSEvent

    @property
    def dt(self):
        return self._simulator.dt

    @property
    def _timestep(self):
        return self._simulator._timestep

    @property
    def size(self):
        return self.num_neurons

    @property
    def size_in(self):
        return self.num_external

    @property
    def state(self):
        return self._simulator.state

    @property
    def stddev_mismatch(self):
        return self._stddev_mismatch

    @property
    def mismatch_factors(self):
        return self._mismatch_factors


# Functions:
# - Different synapse types?
# - Adaptation?
# - NMDA synapses?
# - BUF_P???
# - Syn-thresholds???
# Limitations
