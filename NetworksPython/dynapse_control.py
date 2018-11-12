# ----
# dynapse_control.py - Class to interface the DynapSE chip
# ----

### --- Imports

import numpy as np
from warnings import warn
from typing import Tuple, List, Optional, Union
import time
from .timeseries import TSEvent

# - Programmatic imports for CtxCtl
try:
    import CtxDynapse
    import NeuronNeuronConnector

except ModuleNotFoundError:
    # - Try with RPyC
    try:
        import rpyc
        conn = rpyc.classic.connect('localhost', '1300')
        CtxDynapse = conn.modules.CtxDynapse
        NeuronNeuronConnector = conn.modules.NeuronNeuronConnector
        print("dynapse_control: RPyC connection established.")

    except:
        # - Raise an ImportError (so that the smart __init__.py module can skip over missing CtxCtl
        raise ImportError

# - Imports from ctxCTL
SynapseTypes = CtxDynapse.DynapseCamType
dynapse = CtxDynapse.dynapse
DynapseFpgaSpikeGen = CtxDynapse.DynapseFpgaSpikeGen
DynapsePoissonGen = CtxDynapse.DynapsePoissonGen
DynapseNeuron = CtxDynapse.DynapseNeuron
VirtualNeuron = CtxDynapse.VirtualNeuron
BufferedEventFilter = CtxDynapse.BufferedEventFilter
FpgaSpikeEvent = CtxDynapse.FpgaSpikeEvent

print("dynapse_control: CtxDynapse modules loaded.")


### --- Parameters
# - Fix (hardware)
FPGA_EVENT_LIMIT = int(2 ** 19 - 1)  # Max. number of events that can be sent to FPGA
FPGA_ISI_LIMIT = int(2 ** 16 - 1)  # Max. number of timesteps for single inter-spike interval between FPGA events
FPGA_TIMESTEP = 1. / 9. * 1e-7  # Internal clock of FPGA, 11.111...ns
CORE_DIMENSIONS = (16, 16)  # Numbers of neurons in core (rows, columns)
# - Default values, can be changed
DEF_FPGA_ISI_BASE = 2e-5  # Default timestep between events sent to FPGA
DEF_FPGA_ISI_MULTIPLIER = int(np.round(DEF_FPGA_ISI_BASE / FPGA_TIMESTEP))

# MOVE THIS TO LAYER #
# - Default maximum numbers of time steps for a single evolution batch
#   Assuming one input event after the maximum ISI - This is the maximally possible
#   value. In practice there will be more events per time. Therefore the this value
#   does not guarantee that the complete input batch fits onto the fpga
nDefaultMaxNumTimeSteps = int(FPGA_EVENT_LIMIT * (2 ** 16 - 1))


### --- Utility functions

def generate_fpga_event(nTargetChipID, nTargetCoreMask, nNeuronID, nISI) -> FpgaSpikeEvent:
    event = FpgaSpikeEvent()
    event.target_chip = nTargetChipID
    event.core_mask = nTargetCoreMask
    event.neuron_id = nNeuronID
    event.isi = nISI
    return event

def neurons_to_channels(
    lNeurons: List, lLayerNeurons: List
) -> np.ndarray:
    """
    neurons_to_channels - Convert a list of neurons into layer channel indices

    :param lNeurons:        List Lx0 to match against layer neurons
    :param lLayerNeurons:   List Nx0 HW neurons corresponding to each channel index
    :return:                np.ndarray[int] Lx0 channel indices corresponding to each neuron in lNeurons
    """
    # - Initialise list to return
    lChannelIndices = []

    for neurTest in lNeurons:
        try:
            nChannelIndex = lLayerNeurons.index(neurTest)
        except ValueError:
            warn("Unexpected neuron ID {}".format(neurTest.get_id()))
            nChannelIndex = np.nan

        # - Append discovered index
        lChannelIndices.append(nChannelIndex)

    # - Convert to numpy array
    return np.array(lChannelIndices)

def connectivity_matrix_to_prepost_lists(
    mnW: np.ndarray
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    connectivity_matrix_to_prepost_lists - Convert a matrix into a set of pre-post connectivity lists

    :param mnW: ndarray[int]    Matrix[pre, post] containing integer numbers of synaptic connections
    :return:    (vnPreE,
                 vnPostE,
                 vnPreI,
                 vnPostI)       Each ndarray(int), containing a single pre- and post- synaptic partner.
                                    vnPreE and vnPostE together define excitatory connections
                                    vnPreI and vnPostI together define inhibitory connections
    """
    # - Get lists of pre and post-synaptic IDs
    vnPreECompressed, vnPostECompressed = np.nonzero(mnW > 0)
    vnPreICompressed, vnPostICompressed = np.nonzero(mnW < 0)

    # - Preallocate connection lists
    vnPreE = []
    vnPostE = []
    vnPreI = []
    vnPostI = []

    # - Loop over lists, appending when necessary
    for nPre, nPost in zip(vnPreECompressed, vnPostECompressed):
        for _ in range(mnW[nPre, nPost]):
            vnPreE.append(nPre)
            vnPostE.append(nPost)

    # - Loop over lists, appending when necessary
    for nPre, nPost in zip(vnPreICompressed, vnPostICompressed):
        for _ in range(np.abs(mnW[nPre, nPost])):
            vnPreI.append(nPre)
            vnPostI.append(nPost)

    # - Return augmented lists
    return vnPreE, vnPostE, vnPreI, vnPostI

def rectangular_neuron_arrangement(
    nFirstNeuron: int,
    nNumNeurons: int,
    nWidth: int,
    tupCoreDimensions: tuple=CORE_DIMENSIONS,
) -> List[int]:
    """
    rectangular_neuron_arrangement: return neurons that form a rectangle on the chip
                                    with nFirstNeuron as upper left corner and width
                                    nWidth. (Last row may not be full). Neurons
                                    have to fit onto single core.
    :param nFirstNeuron:  int ID of neuron that is in the upper left corner of the rectangle
    :param nNumNeurons:   int Number of neurons in the rectangle
    :param nWidth:        int Width of the rectangle (in neurons) 
    :param tupCoreDimensions:  tuple (number of neuron rows in core, number of neurons in row)
    :return lNeuronIDs:   list 1D array of IDs of neurons that from the rectangle.
    """
    nHeightCore, nWidthCore = tupCoreDimensions
    nNumRows = int(np.ceil(nNumNeurons / nWidth))
    nFirstRow = int(np.floor(nFirstNeuron / nWidthCore))
    nFirstCol = nFirstNeuron % nWidthCore
    # - Make sure rectangle fits on single core
    assert (
        nFirstCol + nWidth <= nWidthCore  # not too wide
        and nFirstRow % nHeightCore + nNumRows <= nHeightCore  # not too high
    ), "Rectangle does not fit onto single core."
    lNeuronIDs = [
        (nFirstRow + nRow) * nWidthCore + (nFirstCol + nID)
        for nRow in range(nNumRows)
        for nID in range(nWidth)
    ]
    return lNeuronIDs

def generate_event_raster(
    lEvents: list,
    tDuration: float,
    vnNeuronIDs: list,
) -> np.ndarray:
    """
    generate_event_raster - Generate a boolean spike raster of a list of events with timestep 0.001ms
    :param lEvents:         list of Spike objects
    :param tDuration:       float Overall time covered by the raster
    :param vnNeuronIDs:     array-like of neuron IDs corresponding to events.
    
    :return:
        mbEventRaster   np.ndarray  - Boolean event raster
    """
    vnNeuronIDs = list(vnNeuronIDs)        
    # - Extract event timestamps and neuron IDs
    tupTimestamps, tupEventNeuronIDs = zip(*((event.timestamp, event.neuron.get_id()) for event in lEvents))
    # - Event times in microseconds
    vnEventTimes = np.array(tupTimestamps, int)
    # - Set time of first event to 0
    vnEventTimes -= vnEventTimes[0]
    # - Convert neuron IDs of event to index starting from 0
    viEventIndices = np.array([vnNeuronIDs.index(nNeuronID) for nNeuronID in tupEventNeuronIDs])
    # - Convert events to binary raster
    nTimeSteps = int(np.ceil(tDuration*1e6))
    mbEventRaster = np.zeros((nTimeSteps, np.size(viEventIndices)), bool)
    mbEventRaster[vnEventTimes, viEventIndices] = True
    return mbEventRaster

def evaluate_firing_rates(
    lEvents: list,
    tDuration: float,
    vnNeuronIDs: Optional[list] = None,
) -> (np.ndarray, float, float):
    """
    evaluate_firing_rates - Determine the neuron-wise firing rates from a
                            list of events. Calculate mean, max and min.
    :param lEvents:         list of Spike objects
    :param tDuration:       float Time over which rates are normalized
    :param vnNeuronIDs:     array-like of neuron IDs corresponding to events.
                            If None, vnNeuronIDs will consists of to the neurons
                            corresponding to the events in lEvents.

    :return:
        vfFiringRates  np.ndarray - Each neuron's firing rate
        fMeanRate      float - Average firing rate over all neurons
        fMaxRate       float - Highest firing rate of all neurons
        fMinRate       float - Lowest firing rate of all neurons            
    """
    # - Extract event timestamps and neuron IDs
    tupTimestamps, tupEventNeuronIDs = zip(*((event.timestamp, event.neuron.get_id()) for event in lEvents))
    
    # - Count events for each neuron
    viUniqueEventIDs, vnEventCounts = np.unique(tupEventNeuronIDs, return_counts=True)
    
    if vnNeuronIDs is None:
        # - lNeuronIDs as list of neurons that have spiked
        vnNeuronIDs = viUniqueEventIDs
    
    # - Neurons that have not spiked
    viNoEvents = (np.asarray(vnNeuronIDs))[np.isin(vnNeuronIDs, viUniqueEventIDs) == False]
    
    # - Count events
    viUniqueEventIDs = np.r_[viUniqueEventIDs, viNoEvents]
    vnEventCounts = np.r_[vnEventCounts, np.zeros(viNoEvents.size)]

    # - Sort event counts to same order as in vnNeuronIDs
    liUniqueEventIDs = list(viUniqueEventIDs)
    lSort = [liUniqueEventIDs.index(nNeuronID) for nNeuronID in vnNeuronIDs]
    vnEventCounts = vnEventCounts[lSort]
    vfFiringRates = vnEventCounts / tDuration

    # - Calculate mean, max and min rates
    fMeanRate = np.size(lEvents) / tDuration / len(vnNeuronIDs)
    print("Mean firing rate: {} Hz".format(fMeanRate))
    iMaxRateNeuron = np.argmax(vnEventCounts)
    fMaxRate = vfFiringRates[iMaxRateNeuron]
    print("Maximum firing rate: {} Hz (neuron {})".format(fMaxRate, vnNeuronIDs[iMaxRateNeuron]))
    iMinRateNeuron = np.argmin(vnEventCounts)
    fMinRate = vfFiringRates[iMinRateNeuron]
    print("Minimum firing rate: {} Hz (neuron {})".format(fMinRate, vnNeuronIDs[iMinRateNeuron]))

    return vfFiringRates, fMeanRate, fMaxRate, fMinRate


class DynapseControl():

    _nFpgaEventLimit = FPGA_EVENT_LIMIT
    _nFpgaIsiLimit = FPGA_ISI_LIMIT
    _tFpgaTimestep = FPGA_TIMESTEP

    def __init__(
        self, tFpgaIsiBase: float=DEF_FPGA_ISI_BASE, lClearChips: Optional[list]=None
    ):
        """
        DynapseControl - Class for interfacing DynapSE

        :param tFpgaIsiBase:  float  Time step for inter-spike intervalls when
                                     sending events to FPGA
        :param lClearChips:    list or None  IDs of chips where configurations
                                             should be cleared.
        """

        print("DynapseControl: Initializing DynapSE")
       
        # - Chip model and virtual model
        self.model = CtxDynapse.model
        self.virtualModel = CtxDynapse.VirtualModel()

        # - dynapse object from CtxDynapse
        self.dynapse = dynapse

        ## -- Modules for sending input to FPGA
        lFPGAModules = self.model.get_fpga_modules()

        # - Find a spike generator module
        vbIsSpikeGenModule = [isinstance(m, DynapseFpgaSpikeGen) for m in lFPGAModules]
        if not np.any(vbIsSpikeGenModule):
            # There is no spike generator, so we can't use this Python layer on the HW
            raise ModuleNotFoundError(
                "DynapseControl: An `fpgaSpikeGen` module is required to use the DynapSE layer."
            )
        else:
            # Get first spike generator module
            self.fpgaSpikeGen = lFPGAModules[np.argwhere(vbIsSpikeGenModule)[0][0]]
            print("DynapseControl: Spike generator module ready.")

        # - Find a poisson spike generator module
        vbIsPoissonGenModule = [isinstance(m, DynapsePoissonGen) for m in lFPGAModules]
        if np.any(vbIsPoissonGenModule):
            self.fpgaPoissonGen = lFPGAModules[np.argwhere(vbIsPoissonGenModule)[0][0]]
        else:
            warn("DynapseControl: Could not find poisson generator module (DynapsePoissonGen).")

        # - Get all neurons
        self.vHWNeurons = np.asarray(self.model.get_neurons())
        self.vShadowNeurons = np.asarray(self.model.get_shadow_state_neurons())
        self.vVirtualNeurons = np.asarray(self.virtualModel.get_neurons())

        # - Initialise neuron allocation
        self.clear_neuron_assignments(True, True)

        print("DynapseControl: Neurons initialized.")

        # - Get a connector object
        self.dcNeuronConnector = NeuronNeuronConnector.DynapseConnector()
        print("DynapseControl: Neuron connector initialized")

        # - Wipe configuration
        self.clear_chips(lClearChips)

        ## -- Initialize Fpga spike generator
        self.tFpgaIsiBase = tFpgaIsiBase
        self.fpgaSpikeGen.set_repeat_mode(False)
        self.fpgaSpikeGen.set_variable_isi(True)
        self.fpgaSpikeGen.set_base_addr(0)
        print("DynapseControl: FPGA spike generator prepared.")

        print("DynapseControl ready.")

    def clear_chips(self, lClearChips: Optional[list]=None):
        """
        clear_cams - Clear the CAM and SRAM cells of chips defined in lClearCam.
        :param lClearChips:  list or None  IDs of chips where configurations 
                                           should be cleared.
        """
        if lClearChips is None:
            return
        elif isinstance(lClearChips, int):
            lClearChips = (lClearChips, )
        for nChip in lClearChips:
            print("DynapseControl: Clearing chip {}.".format(nChip))
            # - Clear SRAMs
            dynapse.clear_cam(int(nChip))
            print("\t CAMs cleared.")
            # - Clear SRAMs
            print("\t SRAMs cleared.")
            dynapse.clear_sram(int(nChip))
            # - Reset neuron weights in model
            for neuron in self.vShadowNeurons[nChip*1024: (nChip+1)*1024]:
                vSrams = neuron.get_srams()
                for iSramIndex in range(1, 4):
                    vSrams[iSramIndex].set_target_chip_id(0)
                    vSrams[iSramIndex].set_virtual_core_id(0)
                    vSrams[iSramIndex].set_used(False)
                    vSrams[iSramIndex].set_core_mask(0)
                for cam in neuron.get_cams():
                    cam.set_pre_neuron_id(0)
                    cam.set_pre_neuron_core_id(0)
            print("\t Model neuron weights have been reset.")
        print("DynapseControl: Chips cleared.")


    ### --- Neuron allocation and connections

    def clear_neuron_assignments(self, bHardware: bool=True, bVirtual: bool=True):
        """
        clear_neuron_assignments - Mark neurons as free again.
        :param bHardware:   Mark all hardware neurons as free (except neuron 0 of each chip)
        :param bVirtual:    Mark all virtual neurons as free (except neuron 0)
        """
        if bHardware:
            # - Hardware neurons
            self.vbFreeHWNeurons = np.ones(self.vHWNeurons.size, bool)
            # Do not use hardware neurons with ID 0 and core ID 0 (first of each core)
            self.vbFreeHWNeurons[0::1024] = False
            print("DynapseControl: {} hardware neurons available.".format(np.sum(self.vbFreeHWNeurons)))
        if bVirtual:
            # - Virtual neurons
            self.vbFreeVirtualNeurons = np.ones(self.vVirtualNeurons.size, bool)
            # Do not use virtual neuron 0
            self.vbFreeVirtualNeurons[0] = False
            print("DynapseControl: {} virtual neurons available.".format(np.sum(self.vbFreeVirtualNeurons)))        

    def allocate_hw_neurons(self, vnNeuronIDs: Union[int, np.ndarray]) -> (np.ndarray, np.ndarray):
        """
        allocate_hw_neurons - Return a list of neurons that may be used.
                              These are guaranteed not to already be assigned.

        :param vnNeuronIDs:  int or np.ndarray    The number of neurons requested or IDs of requested neurons
        :return:             list                 A list of neurons that may be used
        """

        ## -- Choose neurons
        if isinstance(vnNeuronIDs, int):
            # - Choose first available neurons
            nNumNeurons = vnNeuronIDs
            # - Are there sufficient unallocated neurons?
            if np.sum(self.vbFreeHWNeurons) < nNumNeurons:
                raise MemoryError(
                    "Insufficient unallocated neurons available. {} requested.".format(
                        nNumNeurons
                    )
                )
            # - Pick the first available neurons
            vnNeuronsToAllocate = np.nonzero(self.vbFreeHWNeurons)[0][:nNumNeurons]

        else:
            # - Choose neurons defined in vnNeuronIDs
            vnNeuronsToAllocate = np.array(vnNeuronIDs).flatten()
            # - Make sure neurons are available
            if (self.vbFreeHWNeurons[vnNeuronsToAllocate] == False).any():
                raise MemoryError(
                    "{} of the requested neurons are already allocated.".format(
                        np.sum(
                            self.vbFreeHWNeurons[vnNeuronsToAllocate] == False
                        )
                    )
                )

        # - Mark these neurons as allocated
        self.vbFreeHWNeurons[vnNeuronsToAllocate] = False

        # - Prevent allocation of virtual neurons with same ID as allocated hardware neurons
        vnInputNeuronOverlap = vnNeuronsToAllocate[
            vnNeuronsToAllocate < np.size(self.vbFreeVirtualNeurons)
        ]
        self.vbFreeVirtualNeurons[vnInputNeuronOverlap] = False

        # - Return these allocated neurons
        return (
            self.vHWNeurons[vnNeuronsToAllocate],
            self.vShadowNeurons[vnNeuronsToAllocate]
        )

    def allocate_virtual_neurons(self, vnNeuronIDs: Union[int, np.ndarray]) -> np.ndarray:
        """
        allocate_virtual_neurons - Return a list of neurons that may be used.
                                   These are guaranteed not to already be assigned.

        :param vnNeuronIDs:  int or np.ndarray    The number of neurons requested or IDs of requested neurons
        :return:             list    A list of neurons that may be used
        """
        if isinstance(vnNeuronIDs, int):
            nNumNeurons = vnNeuronIDs
            # - Are there sufficient unallocated neurons?
            if np.sum(self.vbFreeVirtualNeurons) < nNumNeurons:
                raise MemoryError(
                    "Insufficient unallocated neurons available. {}".format(nNumNeurons)
                    + " requested."
                )
            # - Pick the first available neurons
            vnNeuronsToAllocate = np.nonzero(self.vbFreeVirtualNeurons)[0][
                :nNumNeurons
            ]

        else:
            vnNeuronsToAllocate = np.array(vnNeuronIDs).flatten()
            # - Make sure neurons are available
            if (self.vbFreeVirtualNeurons[vnNeuronsToAllocate] == False).any():
                raise MemoryError(
                    "{} of the requested neurons are already allocated.".format(
                        np.sum(
                            self.vbFreeVirtualNeurons[vnNeuronsToAllocate]
                            == False
                        )
                    )
                )

        # - Mark these as allocated
        self.vbFreeVirtualNeurons[vnNeuronsToAllocate] = False
        # - Prevent allocation of hardware neurons with same ID as allocated virtual neurons
        self.vbFreeHWNeurons[vnNeuronsToAllocate] = False

        # - Return these neurons
        return self.vVirtualNeurons[vnNeuronsToAllocate]

    def connect_to_virtual(
        self,
        vnVirtualNeuronIDs: Union[int, np.ndarray],
        vnNeuronIDs: Union[int, np.ndarray],
        lSynapseTypes: List
    ):
        """
        conncect_to_virtual - Connect a group of hardware neurons or
                              single hardware neuron to a groupd of
                              virtual neurons (1 to 1) or to a single
                              virtual neuron.
        :param vnVirtualNeuronIDs:   np.ndarray  IDs of virtual neurons
        :param vnNeuronIDs:          np.ndarray  IDs of hardware neurons
        :param lSynapseTypes:        list        Types of the synapses
        """

        # - Handle single neurons
        if np.size(vnVirtualNeuronIDs) == 1:
            vnVirtualNeuronIDs = np.repeat(vnVirtualNeuronIDs, np.size(vnNeuronIDs))
        if np.size(vnNeuronIDs) == 1:
            vnNeuronIDs = np.repeat(vnNeuronIDs, np.size(vnVirtualNeuronIDs))
        if np.size(lSynapseTypes) == 1:
            lSynapseTypes = list(np.repeat(lSynapseTypes, np.size(vnNeuronIDs)))
        else:
            lSynapseTypes = list(lSynapseTypes)

        # - Get neurons that are to be connected
        lPreNeurons = list(self.vVirtualNeurons[vnVirtualNeuronIDs])
        lPostNeurons = list(self.vShadowNeurons[vnNeuronIDs])
        
        # - Set connections
        self.dcNeuronConnector.add_connection_from_list(
            lPreNeurons,
            lPostNeurons,
            lSynapseTypes,
        )
        print("DynapseControl: Setting up {} connections".format(np.size(vnNeuronIDs)))
        self.model.apply_diff_state()
        print("DynapseControl: Connections set")

    def remove_all_connections_to(self, vnNeuronIDs, bApplyDiff: bool=True):
        """
        remove_all_connections_to - Remove all presynaptic connections
                                    to neurons defined in vnNeuronIDs
        :param vnNeuronIDs:     np.ndarray IDs of neurons whose presynaptic
                                           connections should be removed
        :param bApplyDiff:      bool If False do not apply the changes to
                                     chip but only to shadow states of the
                                     neurons. Useful if new connections are
                                     going to be added to the given neurons.
        """     
        # - Make sure neurons vnNeuronIDs is an array
        vnNeuronIDs = np.asarray(vnNeuronIDs)

        # - Reset neuron weights in model
        for neuron in self.vShadowNeurons[vnNeuronIDs]:
            viSrams = neuron.get_srams()
            for iSramIndex in range(1, 4):
                viSrams[iSramIndex].set_target_chip_id(0)
                viSrams[iSramIndex].set_virtual_core_id(0)
                viSrams[iSramIndex].set_used(False)
                viSrams[iSramIndex].set_core_mask(0)
            viCams = neuron.get_cams()
            for cam in viCams:
                cam.set_pre_neuron_id(0)
                cam.set_pre_neuron_core_id(0)
        print("DynapseControl: Shadow state neuron weights have been reset")
        if bApplyDiff:
            # - Apply changes to the connections on chip
            self.model.apply_diff_state()
            print("DynapseControl: New state has been applied to the hardware")


    ### --- Stimulation and event generation

    def TSEvent_to_spike_list(
        self,
        tsSeries: TSEvent,
        vnNeuronIDs: np.ndarray,
        nTargetCoreMask: int = 1,
        nTargetChipID: int = 0,
    ) -> List:
        """
        TSEvent_to_spike_list - Convert a TSEvent object to a ctxctl spike list

        :param tsSeries:        TSEvent      Time series of events to send as input
        :param vnNeuronIDs:     ArrayLike    IDs of neurons that should appear as sources of the events
        :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
        :param nTargetChipID:   int          ID of target chip
        :return:                list of FpgaSpikeEvent objects
        """
        # - Check that the number of channels is the same between time series and list of neurons
        assert tsSeries.nNumChannels <= np.size(
            lNeurons
        ), "`tsSeries` contains more channels than the number of neurons in `lNeurons`."

        if np.size(vnNeuronIDs == 1):
            # - Make sure vnNeuronIDs is iterable
            vnNeuronIDs = (vnNeuronIDs,)

        # - Get events from this time series
        vtTimes, vnChannels, _ = tsSeries.find()

        # - Convert to ISIs
        tStartTime = tsSeries.tStart
        vtISIs = np.diff(np.r_[tStartTime, vtTimes])
        vnDiscreteISIs = (np.round(vtISIs / self.tFpgaIsiBase)).astype("int")

        # - Convert each event to an FpgaSpikeEvent
        lEvents = [
            generate_fpga_event(
                nTargetChipID, nTargetCoreMask, vnNeuronIDs[nChannel], nISI
            )
            for nChannel, nISI in zip(vnChannels, vnDiscreteISIs)
        ]

        # - Return a list of events
        return lEvents

    def arrays_to_spike_list(
        vnTimeSteps: np.ndarray,
        vnChannels: np.ndarray,
        vnNeuronIDs: np.ndarray,
        nTSStart: int = 0,
        nTargetCoreMask: int = 1,
        nTargetChipID: int = 0,
    ) -> List:
        """
        arrays_to_spike_list - Convert an array of input time steps and an an array
                               of event channels to a ctxctl spike list

        :param vnTimeSteps:     np.ndarray   Event time steps (Using FPGA time base)
        :param vnChannels:      np.ndarray   Event time steps (Using FPGA time base)
        :param vnNeuronIDs:     ArrayLike    IDs of neurons that should appear as sources of the events
        :param nTSStart:        int          Time step at which to start
        :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
        :param nTargetChipID:   int          ID of target chip
        :return:                list of FpgaSpikeEvent objects
        """
        # - Ignore data that comes before nTSStart
        vnTimeSteps = vnTimeSteps[vnTimeSteps >= nTSStart]
        vnChannels = vnChannels[vnTimeSteps >= nTSStart]

        # - Check that the number of channels is the same between time series and list of neurons
        assert np.amax(vnChannels) <= np.size(
            vnNeuronIDs
        ), "`vnChannels` contains more channels than the number of neurons in `vnNeuronIDs`."

        if np.size(vnNeuronIDs == 1):
            # - Make sure vnNeuronIDs is iterable
            vnNeuronIDs = (vnNeuronIDs,)

        # - Convert to ISIs
        vnDiscreteISIs = np.diff(np.r_[nTSStart, vnTimeSteps])

        # - Convert each event to an FpgaSpikeEvent
        lEvents = [
            generate_fpga_event(
                nTargetChipID, nTargetCoreMask, vnNeuronIDs[nChannel], nISI
            )
            for nChannel, nISI in zip(vnChannels, vnDiscreteISIs)
        ]

        # - Return a list of events
        return lEvents

    def start_cont_stim(
        self,
        fFrequency: float,
        nNeuronID: int,
        nChipID: int=0,
        nCoreMask: int=15,
    ):
        """
        start_cont_stim - Start sending events with fixed frequency.
                          FPGA repeat mode will be set True.
        :param fFrequency:  float  Frequency at which events are sent
        :param nNeuronID:   int  Event neuron ID(s)
        :param nChipID:     int  Target chip ID
        :param nCoreMask:   int  Target core mask
        """
        
        # - Set FPGA to repeating mode
        self.fpgaSpikeGen.set_repeat_mode(True)

        # - Interspike interval
        tISI = 1. / fFrequency
        # - ISI in units of fpga time step
        nISIfpga = int(np.round(tISI / self.tFpgaIsiBase))

        # - Handle integers for vnNeuronIDs
        if isinstance(vnNeuronIDs, int):
            vnNeuronIDs = [vnNeuronIDs]

        # - Generate events
        # List for events to be sent to fpga
        lEvents = [
            generate_fpga_event(nChipID, nCoreMask, nID, nISIfpga)
        ]

        self.fpgaSpikeGen.preload_stimulus(lEvents)
        print("DynapseControl: Stimulus prepared with {} Hz".format(
            1. / (nISIfpga * self.tFpgaIsiBase))
        )
        
        # - Start stimulation
        self.fpgaSpikeGen.start()
        print("DynapseControl: Stimulation started")

    def stop_stim(self, bClearFilter: bool=False):
        """
        stop_stim - Stop stimulation with FGPA spke generator. 
                    FPGA repeat mode will be set False.
        :param bStopRecording:  bool  Clear buffered event filter if present.
        """
        # - Stop stimulation
        self.fpgaSpikeGen.stop()
        # - Set default FPGA settings
        self.fpgaSpikeGen.set_repeat_mode(False)
        print("DynapseControl: Stimulation stopped")
        if bClearFilter:
            self.clear_buffered_event_filter()

    def start_poisson_stim(
        self,
        vfFrequencies: Union[int, np.ndarray],
        vnNeuronIDs: Union[int, np.ndarray],
        nChipID: int = 0,
    ):
        """
        start_poisson_stim - Start generated events by poisson processes and send them.
        :param vfFrequency: int or array-like  Frequencies of poisson processes
        :param vnNeuronIDs: int or array-like  Event neuron ID(s)
        :param nChipID:     int  Target chip ID
        """
        
        # - Handle single values for frequencies and neurons
        if np.size(vnNeuronIDs) == 1:
            vnNeuronIDs = np.repeat(vnNeuronIDs, np.size(vfFrequencies))
        if np.size(vfFrequencies) == 1:
            vfFrequencies = np.repeat(vfFrequencies, np.size(vnNeuronIDs))
        else:
            assert np.size(vfFrequencies) == np.size(vnNeuronIDs), \
                "DynapseControl: Length of `vfFrequencies must be same as length of `vnNeuronIDs` or 1."

        # - Set firing rates for selected neurons
        for fFreq, nNeuronID in zip(vfFrequencies, vnNeuronIDs):
            self.fpgaPoissonGen.write_poisson_rate_hz(nNeuronID, fFreq)

        # - Set chip ID
        self.fpgaPoissonGen.set_target_chip_id(nChipID)

        print("DynapseControl: Poisson stimuli prepared for chip {}.".format(nChipID))
        
        # - Start stimulation
        self.fpgaPoissonGen.start()
        print("DynapseControl: Poisson rate stimulation started")

    def stop_poisson_stim(self):
        """
        stop_stim - Stop stimulation with FGPA poisson generator.
                    Does not stop any event recording.
        """
        self.fpgaPoissonGen.stop()
        print("DynapseControl: Poisson rate stimulation stopped")

    def reset_poisson_rates(self):
        """reset_poisson_rates - Set all firing rates of poisson generator to 0."""
        for i in range(1024):
            self.fpgaPoissonGen.write_poisson_rate_hz(i, 0)
        print("DynapseControl: Firing rates for poisson generator have been set to 0.")

    def send_pulse(
            self,
            tWidth: float=0.1,
            fFreq: float=1000,
            tRecordTime: float=3,
            nInputNeuronID: int=0,
            vnRecordNeuronIDs: Union[int, np.ndarray]=0,
            nTargetCoreMask: int=15,
            nTargetChipID: int=0,
        ) -> TSEvent:
        """
        send_pulse - Send a pulse of periodic input events to the chip.
                     Return a TSEvent wih the recorded hardware activity.
        :param tWidth:              float  Duration of the input pulse
        :param tFreq:               float  Frequency of the events that constitute the pulse
        :param nInputNeuronID:      int    ID of input neuron
        :param vnRecordNeuronIDs:   array-like  ID(s) of neuron(s) to be recorded
        :param tRecordTime:         float  Duration of the recording (including stimulus)
        :param nChipID:     int  Target chip ID
        :param nCoreMask:   int  Target core mask

        :return:
            TSEvent  Recorded events
        """
        # - Prepare input events
        # Actual input time steps
        vnTimeSteps = np.floor(np.arange(0, tWidth, 1./fFreq) / self.tFpgaIsiBase).astype(int)
        # Add dummy events at end to avoid repeated stimulation due to "2-trigger-bug"
        tISILimit = self.nFpgaIsiLimit * self.tFpgaIsiBase
        nAdd = int(np.ceil(tRecordTime / tISILimit))
        vnTimeSteps = np.r_[vnTimeSteps, vnTimeSteps[-1] + np.arange(1, nAdd+1)*self.nFpgaIsiLimit]
        
        lEvents = arrays_to_spike_list(
            vnTimeSteps=vnTimeSteps,
            vnChannels=np.repeat(nInputNeuronID, vnTimeSteps.size),
            nTSStart=0,
            lNeurons=self.vHWNeurons,
            nTargetCoreMask=nTargetCoreMask,
            nTargetChipID=nTargetChipID,        
        )
        # - Do not send dummy events to any core
        for event in lEvents[-nAdd:]:
            event.core_mask = 0
        print("DynapseControl: Stimulus prepared")
        
        # - Prepare FPGA
        self.fpgaSpikeGen.set_repeat_mode(False)
        self.fpgaSpikeGen.preload_stimulus(lEvents)
        print("DynapseControl: Stimulus preloaded.")

        # - Set up event recording
        if np.size(vnRecordNeuronIDs) == 1:
            vnRecordNeuronIDs = np.array(vnRecordNeuronIDs)
        lnRecordNeuronIDs = list(vnRecordNeuronIDs)
        oFilter = BufferedEventFilter(self.model, lnRecordNeuronIDs)
        print("DynapseControl: Event filter ready.")

        # - Stimulate / record
        print("DynapseControl: Starting stimulation.")
        self.fpgaSpikeGen.start()

        # - Run stimulation and record
        time.sleep(tRecordTime)

        # - Stop stimulation and clear filter to stop recording events
        self.fpgaSpikeGen.stop()
        oFilter.clear()
        print("DynapseControl: Stimulation ended.")

        lEvents = list(oFilter.get_events())
        lTrigger = list(oFilter.get_special_event_timestamps())
        print(
            "DynapseControl: Recorded {} events and {} trigger events".format(len(lEvents), len(lTrigger))
        )

        # - Extract monitored event channels and timestamps
        vnChannels = neurons_to_channels(
            [e.neuron for e in lEvents], lnRecordNeuronIDs
        )
        vtTimeTrace = np.array([e.timestamp for e in lEvents]) * 1e-6

        # - Locate synchronisation timestamp
        tStartTrigger = lTrigger[0] * 1e-6
        iStartIndex = np.searchsorted(vtTimeTrace, tStartTrigger)
        vnChannels = vnChannels[iStartIndex:]
        vtTimeTrace = vtTimeTrace[iStartIndex:] - tStartTrigger
        print("DynapseControl: Extracted event data")

        return TSEvent(
            vtTimeTrace,
            vnChannels,
            tStart=0,
            tStop=tRecordTime,
            nNumChannels=len(lnRecordNeuronIDs)
        )

    def send_TSEvent(
        self,
        tsSeries,
        vnNeuronIDs,
        nTargetCoreMask: int=15,
        nTargetChipID: int=0,
        bPeriodic=False,
        bRecord=False,
    ):
        """
        send_TSEvent - Extract events from a TSEvent object and send them to FPGA.

        :param tsSeries:        TSEvent      Time series of events to send as input
        :param vnNeuronIDs:     ArrayLike    IDs of neurons that should appear as sources of the events
        :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
        :param nTargetChipID:   int          ID of target chip
        :param bPeriodic:       bool         Repeat the stimulus indefinitely
        :param bRecord:         bool         Set up buffered event filter that records events
                                             from neurons defined in vnNeuronIDs
        """
        lEvents = self.TSEvent_to_spike_list(
            tsSeries,
            vnNeuronIDs=vnNeuronIDs,
            nTargetCoreMask=nTargetCoreMask,
            nTargetChipID=nTargetChipID,        
        )
        print("DynapseControl: Stimulus prepared")
    
        # - Prepare FPGA
        self.fpgaSpikeGen.set_repeat_mode(bPeriodic)
        self.fpgaSpikeGen.preload_stimulus(lEvents)
        print("DynapseControl: Stimulus preloaded.")

        if bRecord:
            self.add_buffered_event_filter(vnNeuronIDs)

        # - Stimulate
        if bPeriodic:
            print(
                "DynapseControl: Starting periodic stimulation with TSEvent `{}`.".format(
                    tsSeries.strName
                )
            )
        else:
            print(
                "DynapseControl: Starting stimulation with TSEvent `{}` for {} s.".format(
                    tsSeries.strName, tsSeries.tDuration
                )
            )
        self.fpgaSpikeGen.start()

    ### --- Tools for tuning and observing activities

    def add_buffered_event_filter(self, vnNeuronIDs):
        """
        add_buffered_event_filter - Add a BufferedEventFilter to record from
                                    neurons defined in vnNeuronIDs
        :param vnNeuronIDs:   array-like  IDs of neurons to be recorded
        :return:
            Reference to selfbufferedfilter, the BufferedEventFilter that has been created.
        """
        # - Convert vnNeuronIDs to list
        if isinstance(vnNeuronIDs, int):
            lnNeuronIDs = range(vnNeuronIDs)
        lnRecordNeuronIDs = list(vnNeuronIDs)
        
        # - Does a filter already exist?
        if hasattr(self, "_bufferedfilter") and self.bufferedfilter is not None:
            self.bufferedfilter.clear()
            self.bufferedfilter.add_ids(lnRecordNeuronIDs)
            print("DynapseControl: Updated existing buffered event filter.")
        else:
            self.bufferedfilter = BufferedEventFilter(self.model, lnNeuronIDs)
            print("DynapseControl: Generated new buffered event filter.")
        
        return bufferedfilter

    def clear_buffered_event_filter(self):
        """ clear_buffered_event_filter - Clear self.bufferedfilter if it exists."""
        if hasattr(self, "_bufferedfilter") and self.bufferedfilter is not None:
            self.bufferedfilter.clear()
            print("DynapseControl: Buffered event filter cleared")
        else:
            warn("DynapseControl: No buffered event filter found.")

    def buffered_events_to_TSEvent(self, vnNeuronIDs=range(4096), tDuration=None):
        """
        buffered_events_to_TSEvent - Fetch events from self.bufferedfilter and 
                                     convert them to a TSEvnet
        :param vnNeuronIDs:     
        """
        # - Fetch events from filter
        lEvents = list(TR_oFilter.get_events())
        lTrigger = list(TR_oFilter.get_special_event_timestamps())
        print(
            "DynapseControl: Recorded {} events and {} trigger events".format(
                len(lEvents), len(lTrigger)
            )
        )

        # - Extract monitored event channels and timestamps
        vnNeuronIDs = np.array(vnNeuronIDs)
        vnChannels = DHW.neurons_to_channels(
            [e.neuron for e in lEvents], list(self._vHWNeurons[vnNeuronIDs])
        )
        # - Remove events that are not from neurons defined in vnNeuronIDs
        vnChannels = vnChannels[np.isnan(vnChannels) == False]
        # - TIme trace
        vtTimeTrace = (
            np.array([e.timestamp for e in lEvents]) * 1e-6
        )[np.isnan(vnChannels) == False]

        # - Locate synchronisation timestamp
        tStartTrigger = lTrigger[0] * 1e-6
        iStartIndex = np.searchsorted(vtTimeTrace, tStartTrigger)
        iEndIndex = (
            None if tDuration is None
            else np.searchsorted(vtTimeTrace, tStartTrigger+tDuration)
        )
        vnChannels = vnChannels[iStartIndex:iEndIndex]
        vtTimeTrace = vtTimeTrace[iStartIndex:iEndIndex] - tStartTrigger
        print("DynapseControl: Extracted event data")

        return TSEvent(
            vtTimeTrace,
            vnChannels,
            tStart=0,
            tStop=tDuration,
            nNumChannels=np.size(vnNeuronIDs)
        )

    def collect_spiking_neurons(self, vnNeuronIDs, tDuration):
        """
        collect_spiking_neurons - Return a list of IDs of neurons that
                                  that spike within tDuration
        :param vnNeuronIDs:   list   IDs of neurons to be observed.
        :param tDuration:     float  How long to wait for spiking neurons 
        :return  lnRecordedNeuronIDs:  list IDs of neurons that have spiked.
        """

        # - Convert vnNeuronIDs to list
        if isinstance(vnNeuronIDs, int):
            lnNeuronIDs = range(vnNeuronIDs)
        lnRecordNeuronIDs = list(vnNeuronIDs)
        
        print("DynapseControl: Collecting IDs of neurons that spike within the next {} seconds".format(tDuration))
        
        # - Filter for recording neurons
        oFilter = BufferedEventFilter(self.model, lnNeuronIDs)
        
        # - Wait and record spiking neurons
        time.sleep(tDuration)
        
        oFilter.clear()
        
        # - Sorted unique list of neurons' IDs that have spiked
        lnRecordedNeuronIDs = sorted(set((event.neuron.get_id() for event in oFilter.get_events())))
        print("DynapseControl: {} neurons spiked: {}".format(len(lnRecordedNeuronIDs), lnRecordedNeuronIDs))

        return lnNeuronIDs

    def silence_hot_neurons(self, vnNeuronIDs, tDuration):
        """
        silence_hot_neurons - Collect IDs of all neurons that spike 
                              within tDuration. Assign them different
                              time constant to silence them.
        :param vnNeuronIDs:  list   IDs of neurons to be observed.
        :param tDuration:    float  How long to wait for spiking neurons 
        """
        # - Neurons that spike within tDuration
        lnHotNeurons = self.collect_spiking_neurons(vnNeuronIDs, tDuration=tDuration)
        # - Silence these neurons by assigning different Tau bias
        for nID in lnHotNeurons:
            self.dynapse.set_tau_2(0, nID)
        print("DynapseControl: Neurons {} have been silenced".format(lnHotNeurons))

    def measure_population_firing_rates(
        self,
        llnPopulationIDs: list,
        tDuration: float,
        bVerbose=False
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        measure_population_firing_rates - Measure the mean, maximum and minimum
                                          firing rates for multiple neuron populatios
        :param llnPopulationIDs:  Array-like of array-like of neuron IDs of each pouplaiton
        :param tDuration:         float  Time over which rates are measured
        :param bVerbose:          bool   Print firing rates of each neuron for each population

        :return:
            vfMeanRates     np.ndarray - Population mean rates
            vfMaxRates     np.ndarray - Population maximum rates
            vfMinRates     np.ndarray - Population minimum rates
        """

        # - Arrays for collecting rates
        nNumPopulations = np.size(llnPopulationIDs)
        vfMeanRates = np.zeros(nNumPopulations)
        vfMaxRates = np.zeros(nNumPopulations)
        vfMinRates = np.zeros(nNumPopulations)

        for i, lnNeuronIDs in enumerate(llnPopulationIDs):
            print("DynapseControl: Population {}".format(i))
            vfFiringRates, vfMeanRates[i], vfMaxRates[i], vfMinRates[i] = self.measure_firing_rates(lnNeuronIDs, tDuration)
            if bVerbose:
                print(vfFiringRates)

        return vfMeanRates, vfMaxRates, vfMinRates

    def measure_firing_rates(
        self,
        vnNeuronIDs: Optional[Union[int, np.ndarray]],
        tDuration: float
    ) -> (np.ndarray, float, float, float):
        """
        measure_firing_rates - Measure the mean, maximum and minimum firing rate
                               for a group of neurons
        :param vnNeuronIDs:     Array-like or int  Neuron IDs to be measured
        :param tDuration:       float  Time over which rates are measured

        :return:
            vfFiringRates  np.ndarray - Each neuron's firing rate
            fMeanRate      float - Average firing rate over all neurons
            fMaxRate       float - Highest firing rate of all neurons
            fMinRate       float - Lowest firing rate of all neurons
        """
        if isinstance(vnNeuronIDs, int):
            vnNeuronIDs = [vnNeuronIDs]
        # - Filter for recording events
        oFilter = BufferedEventFilter(self.model, list(vnNeuronIDs))
        # - Record events for tDuration
        time.sleep(tDuration)
        # - Stop recording
        oFilter.clear()
        # - Retrieve recorded events
        lEvents = list(oFilter.get_events())
        if not lEvents:
            # - Handle empty event lists
            print("DynapseControl: No events recorded")
            return np.zeros(np.size(vnNeuronIDs)), 0, 0, 0
        # - Evaluate non-empty event lists
        return evaluate_firing_rates(lEvents, tDuration, vnNeuronIDs)

    def sweep_freq_measure_rate(
        self,
        vfFreq: list=[1, 10, 20, 50, 100, 200, 500, 1000, 2000],
        tDuration: float=1,
        vnTargetNeuronIDs: Union[int,np.ndarray]=range(128),
        vnInputNeuronIDs: Union[int,np.ndarray]=1,
        nChipID: int=0,
        nCoreMask: int=15,
    ):
        """
        sweep_freq_measure_rate - Stimulate a group of neurons by sweeping
                                  over a list of input frequencies. Measure
                                  their firing rates.
        :param vfFreq:      array-like Stimulus frequencies
        :param tDuration:   float  Stimulus duration for each frequency
        :param vnTargetNeuronIDs: array-like  source neuron IDs
        :param vnInputNeuronIDs:  array-like  IDs of neurons 
        :param nChipID:     int  Target chip ID
        :param nCoreMask:   int  Target core mask

        :return:
            mfFiringRates   np.ndarray -  Matrix containing firing rates for each neuron (axis 1)
                                          for each frequency (axis 0)
            vfMeanRates     np.ndarray - Average firing rates over all neurons for each input frequency
            vfMaxRates      np.ndarray - Highest firing rates of all neurons for each input frequency
            vfMinRates      np.ndarray - Lowest firing rates of all neurons for each input frequency
        """

        # - Arrays for collecting firing rates
        mfFiringRates = np.zeros((np.size(lfFreq), np.size(vnTargetNeuronIDs)))
        vfMeanRates = np.zeros(np.size(lfFreq))
        vfMaxRates = np.zeros(np.size(lfFreq))
        vfMinRates = np.zeros(np.size(lfFreq))

        # - Sweep over frequencies
        for iTrial, fFreq in enumerate(lfFreq):
            print("DynapseControl: Stimulating with {} Hz input".format(fFreq))
            self.start_cont_stim(fFreq, vnInputNeuronIDs, bVirtual)
            mfFiringRates[iTrial, :], vfMeanRates[iTrial], vfMaxRates[iTrial], vfMinRates[iTrial] = (
                self.measure_firing_rates(vnTargetNeuronIDs, tDuration)
            )
            self.stop_stim()

        return mfFiringRates, vfMeanRates, vfMaxRates, vfMinRates

    
    ### - Load and save biases

    @staticmethod
    def load_biases(strFilename):
        """load_biases - Load biases from python file under path strFilename"""
        exec(open(strFilename).read())
        print("DynapseControl: Biases have been loaded from {}.".format(strFilename))
    
    @staticmethod
    def save_biases(strFilename):
        """save_biases - Save biases in python file under path strFilename"""
        PyCtxUtils.save_biases(strFilename)
        print("DynapseControl: Biases have been saved under {}.".format(strFilename))

    def copy_biases(self, nSourceCoreID: int=0, vnTargetCoreIDs: Optional[List[int]]=None):
        """
        copy_biases - Copy biases from one core to one or more other cores.
        :param nSourceCoreID:   int  ID of core from which biases are copied
        :param vnTargetCoreIDs: int or array-like ID(s) of core(s) to which biases are copied
                                If None, will copy to all other neurons
        """

        if vnTargetCoreIDs is None:
            # - Copy biases to all other cores except the source core
            vnTargetCoreIDs = list(range(16))
            vnTargetCoreIDs.remove(nSourceCoreID)
        elif np.size(vnTargetCoreIDs) == 1:
            vnTargetCoreIDs = list(np.array(vnTargetCoreIDs))
        else:
            vnTargetCoreIDs = list(vnTargetCoreIDs)

        # - List of bias groups from all cores
        lBiasgroups = self.model.get_bias_groups()
        sourcebiases = lBiasgroups[nSourceCoreID].get_biases()

        # - Set biases for target cores
        for nTargetCoreID in vnTargetCoreIDs:
            for bias in sourcebiases:
                lBiasgroups[nTargetCoreID].set_bias(bias.bias_name, bias.fine_value, bias.coarse_value)
        print("DynapseControl: Biases copied from core {} to core(s) {}".format(nSourceCoreID, vnTargetCoreIDs))


    ### --- Class properties

    @property
    def synSE(self):
        return SynapseTypes.SLOW_EXC
    @property
    def synSI(self):
        return SynapseTypes.SLOW_INH
    @property
    def synFE(self):
        return SynapseTypes.FAST_EXC
    @property
    def synFI(self):
        return SynapseTypes.FAST_INH

    @property
    def nFpgaEventLimit(self):
        return self._nFpgaEventLimit

    @property
    def nFpgaIsiLimit(self):
        return self._nFpgaIsiLimit

    @property
    def tFpgaTimestep(self):
        return self._tFpgaTimestep

    @property
    def tFpgaIsiBase(self):
        return self._nFpgaIsiMultiplier * self.tFpgaTimestep

    @tFpgaIsiBase.setter
    def tFpgaIsiBase(self, tNewBase):
        if not tNewBase > self.tFpgaTimestep:
            raise ValueError(
                "DynapseControl: `tFpgaTimestep` must be at least {}".format(self.tFpgaTimestep)
            )
        else:
            self._nFpgaIsiMultiplier = int(np.floor(tNewBase / self.tFpgaTimestep))
            self.fpgaSpikeGen.set_isi_multiplier(self._nFpgaIsiMultiplier)