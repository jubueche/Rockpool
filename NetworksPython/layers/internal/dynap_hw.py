# ----
# dynap_hw.py - Implementation of HW FF and Rec layers for DynapSE, via ctxCTL
# ----

from ..layer import Layer
from ...timeseries import TSEvent
from ...dynapse_control import (
    DynapseControl,
    connectivity_matrix_to_prepost_lists,
)

import numpy as np
from warnings import warn
from typing import Tuple, List, Optional, Union
import time


# - Default timestep
DEF_TIMESTEP = 2e-5

# - Absolute tolerance, e.g. for comparing float values
ABS_TOLERANCE = 1e-9

# -- Define the HW layer class for recurrent networks
class RecDynapSE(Layer):
    """
    RecDynapSE - Recurrent layer implemented on DynapSE
    """

    def __init__(
        self,
        mfWIn: np.ndarray,
        mfWRec: np.ndarray,
        vnLayerNeuronIDs: Optional[np.ndarray] = None,
        vnVirtualNeuronIDs: Optional[np.ndarray] = None,
        tDt: Optional[float] = DEF_TIMESTEP,
        fNoiseStd: Optional[float] = None,
        nMaxTrialsPerBatch: Optional[float] = None,
        tMaxBatchDur: Optional[float] = None,
        nMaxNumTimeSteps: Optional[int] = None,
        nMaxEventsPerBatch: Optional[int] = None,
        lInputCoreIDs: List[int] = [0],
        nInputChipID: int = 0,
        controller: DynapseControl=None,
        lClearChips: Optional[list]=[0],
        strName: Optional[str] = "unnamed",
    ):
        """
        RecDynapSE - Recurrent layer implemented on DynapSE

        :param mfWIn:               ndarray[int] MxN matrix of input weights from virtual to hardware neurons
        :param mfWRec:              ndarray[int] NxN matrix of weights between hardware neurons.
                                                 Supplied in units of synaptic connection. Negative elements
                                                 lead to inhibitory synapses
        :param vnLayerNeuronIDs:    ndarray  1D array of IDs of N hardware neurons that are to be used as layer neurons
        :param vnVirtualNeuronIDs:  ndarray  1D array of IDs of M virtual neurons that are to be used as input neurons
        :param tDt:                 float   Time-step.
        :param fNoiseStd:           float   Dummy noise to inject. Not used in layer evolution
        :param nMaxTrialsPerBatch:  int  Maximum number of trials (specified in input timeseries) per batch.
                                         Longer evolution periods will automatically split in smaller batches.
        :param tMaxBatchDur:        float  Maximum duration of single evolution batch.
        :param nMaxNumTimeSteps:    float  Maximum number of time steps of of single evolution batch.
        :param nMaxEventsPerBatch:  float  Maximum number of input events per evolution batch.
        :param lInputCoreIDs:       array-like  IDs of the cores that contain neurons receiving external inputs.
                                                To avoid ID collisions neurons on these cores should not receive inputs
                                                from other neurons.
        :param nInputChipID:        int  ID of the chip with neurons that receive external input.
        :param controller:          DynapseControl object to interface the hardware
        :param lClearChips:         list or None  IDs of chips where configurations should be cleared.
        :param strName:             str     Layer name
        """

        # - Instantiate DynapseControl
        if controller is None:
            if tDt is None:
                raise ValueError("Layer `{}` Either tDt or controller must be provided".format(strName))
            self.controller = DynapseControl(tDt, lClearChips)
        else:
            self.controller = controller
            self.controller.tFpgaIsiBase = tDt
            self.controller.clear_chips(lClearChips)

        # - Check supplied arguments
        if fNoiseStd is not None:
            warn(
                "Layer `{}`: Caution: `fNoiseStd` is ignored during DynapSE layer evolution.".format(
                    strName
                )
            )
        else:
            fNoiseStd = 0.

        assert (
            mfWRec.shape[0] == mfWRec.shape[1]
        ), "Layer `{}`: The recurrent weight matrix `mnWRec` must be square.".format(
            strName
        )

        # - Initialise superclass
        super().__init__(
            mfW=np.asarray(np.round(mfWIn), "int"),
            tDt=tDt,
            fNoiseStd=fNoiseStd,
            strName=strName,
        )
        print("Layer `{}`: Superclass initialized".format(strName))

        # - Check weight matrices
        assert (
            mfWIn.shape[1] == mfWRec.shape[0]
        ), "Layer `{}`: `mnWIn` and `mnWRec` must have compatible shapes: `mnWIn` is MxN, `mnWRec` is NxN.".format(
            strName
        )

        # - Store weight matrices
        self.mfWIn = mfWIn
        self.mfWRec = mfWRec
        # - Record input core mask and chip ID
        self._nInputCoreMask = int(np.sum([2**nID for nID in lInputCoreIDs]))
        self._nInputChipID = nInputChipID
        # - Store evolution batch size limitations
        self.nMaxTrialsPerBatch = nMaxTrialsPerBatch
        self.nMaxEventsPerBatch = (
            self.controller.nFpgaEventLimit if nMaxEventsPerBatch is None
            else nMaxTrialsPerBatch
        )
        if nMaxNumTimeSteps is not None:
            if tMaxBatchDur is not None:
                warn(
                    "Layer `{}`: Caution: If both `nMaxNumTimeSteps` and `tMaxBatchDur` are provided, only `nMaxNumTimeSteps` is considered.".format(
                        strName
                    )
                )
            self.nMaxNumTimeSteps = nMaxNumTimeSteps
        else:
            self.tMaxBatchDur = tMaxBatchDur

        # - Allocate layer neurons
        self._vHWNeurons, self._vShadowNeurons = (
            self.controller.allocate_hw_neurons(self.nSize) if vnLayerNeuronIDs is None
            else self.controller.allocate_hw_neurons(vnLayerNeuronIDs)
        )
        # Make sure number of neurons is correct
        assert self._vHWNeurons.size == self.nSize, (
            "Layer `{}`: `vnLayerNeuronIDs` must be of size {} or None.".format(strName, self.nSize)
        )
        # - Keep list of neuron IDs
        self._vnHWNeuronIDs = np.array([neuron.get_id() for neuron in self._vHWNeurons])
        print("Layer `{}`: Layer neurons allocated".format(strName))

        # - Allocate virtual neurons
        self._vVirtualNeurons = (
            self.controller.allocate_virtual_neurons(self.nSizeIn) if vnVirtualNeuronIDs is None
            else self.controller.allocate_virtual_neurons(vnVirtualNeuronIDs)
        )
        # Make sure number of neurons is correct
        assert self._vVirtualNeurons.size == self.nSizeIn, (
            "Layer `{}`: `vnVirtualNeuronIDs` must be of size {} or None.".format(strName, self.nSizeIn)
        )
        # - Keep list of neuron IDs
        self._vnVirtualNeuronIDs = np.array([neuron.get_neuron_id() for neuron in self._vVirtualNeurons])
        print("Layer `{}`: Virtual neurons allocated".format(strName))
        
        # - Store recurrent weights
        self._mfWRec = np.asarray(np.round(mfWRec), int)

        # - Configure connectivity
        self._compile_weights_and_configure()

        print("Layer `{}` prepared.".format(self.strName))

    def _batch_input_data(
        self, tsInput: TSEvent, nNumTimeSteps: int, bVerbose: bool = False
    ) -> (np.ndarray, int):
        """_batch_data: Generator that returns the data in batches"""
        # - Time points of input trace in discrete layer time base
        vnTSInputEvents = np.floor(tsInput.vtTimeTrace / self.tDt).astype(int)
        # - Make sure evolution is within correct interval
        iStartAll = np.searchsorted(vnTSInputEvents, self._nTimeStep)
        iEndAll = np.searchsorted(vnTSInputEvents, self._nTimeStep + nNumTimeSteps)
        vnTSInputEvents = vnTSInputEvents[iStartAll:iEndAll]
        vnInputChannels = tsInput.vnChannels[iStartAll:iEndAll]
        # vnInputChannels = tsInput.vnChannels

        # - Check whether data for splitting by trial is available
        if hasattr(tsInput, "vtTrialStarts") and self.nMaxTrialsPerBatch is not None:
            ## -- Split by trials
            vnTrialStarts = np.floor(tsInput.vtTrialStarts / self.tDt).astype(int)
            # - Make sure only trials within evolution period are considered
            vnTrialStarts = vnTrialStarts[
                np.logical_and(
                    self._nTimeStep <= vnTrialStarts,
                    vnTrialStarts < self._nTimeStep + nNumTimeSteps,
                )
            ]
            # - Total number of trials
            nNumTrials = vnTrialStarts.size
            # - Array indices of tsInput.vtTimeTrace and tsInput.vnChannels where trials start
            viTrialStartIndices = np.searchsorted(vnTSInputEvents, vnTrialStarts)
            # - Count number of events for each trial (np.r_ to include last trial)
            vnCumulEventsPerTrial = np.r_[viTrialStartIndices, vnTSInputEvents.size]
            vnEventsPerTrial = np.diff(vnCumulEventsPerTrial)

            # - First trial of current batch
            iCurrentTrial = 0
            while iCurrentTrial < nNumTrials:
                # - Cumulated numbers of events per trial for coming trials
                vnCumulNextEvents = (
                    vnCumulEventsPerTrial[iCurrentTrial + 1 :]
                    - vnCumulEventsPerTrial[iCurrentTrial]
                )
                nMaxNumTrialsE = np.searchsorted(
                    vnCumulNextEvents, self.nMaxEventsPerBatch
                )
                if self.nMaxNumTimeSteps is not None:
                    # - Cumulated numbers of time steps per trial for coming trials (np.r_ for including last trial)
                    vnCumulNextTimeSteps = (
                        np.r_[vnTrialStarts[iCurrentTrial + 1 :], nNumTimeSteps]
                        - vnTrialStarts[iCurrentTrial]
                    )
                    # - Maximum number of trials before self.nMaxNumTimeSteps is exceeded
                    nMaxNumTrialsNTS = np.searchsorted(
                        vnCumulNextEvents, self.nMaxNumTimeSteps
                    )
                else:
                    nMaxNumTrialsNTS = np.inf
                # - Number of trials to be used in current batch, considering max. number of trials per batch,
                #   events per batch and (if applies) time steps per batch
                nNumTrialsBatch = min(
                    self.nMaxTrialsPerBatch, nMaxNumTrialsE, nMaxNumTrialsNTS
                )
                assert nNumTrialsBatch > 0, (
                    "Layer `{}`: Cannot continue evolution. ".format(self.strName)
                    + "Either too many timesteps or events in this trial."
                )
                # - Start and end time steps and indices (wrt vnTSInputEvents) of this batch
                nTSStartBatch = vnTrialStarts[iCurrentTrial]
                iStartBatch = viTrialStartIndices[iCurrentTrial]
                try:
                    nTSEndBatch = vnTrialStarts[iCurrentTrial + nNumTrialsBatch]
                    iEndBatch = viTrialStartIndices[
                        iCurrentTrial + nNumTrialsBatch
                    ]
                except IndexError as e:
                    # - If index error occurs because final batch is included
                    if iCurrentTrial + nNumTrialsBatch == viTrialStartIndices.size:
                        iEndBatch = vnTSInputEvents.size
                        nTSEndBatch = nNumTimeSteps + self._nTimeStep
                    else:
                        raise e
                # - Event data to be sent to FPGA
                vnTSInputEventsBatch = vnTSInputEvents[iStartBatch:iEndBatch]
                vnInputChannelsBatch = vnInputChannels[iStartBatch:iEndBatch]
                nNumTimeStepsBatch = nTSEndBatch - nTSStartBatch  # This is not the same as vnTSInputEventsBatch.size as the latter only contains events and not the complete time base
                if bVerbose:
                    nNumEventsBatch = iEndBatch - iStartBatch
                    print(
                        "Layer `{}`: Current batch input: {} s ({} timesteps)".format(
                            self.strName,
                            nNumTimeStepsBatch * self.tDt,
                            nNumTimeStepsBatch,
                        )

                        + ", {} events, trials {} to {} of {}".format(
                            nNumEventsBatch,
                            iCurrentTrial+1,
                            iCurrentTrial+nNumTrialsBatch,
                            nNumTrials,
                        )
                    )
                yield (
                    vnTSInputEventsBatch,
                    vnInputChannelsBatch,
                    nTSStartBatch,
                    nNumTimeStepsBatch * self.tDt,
                )
                iCurrentTrial += nNumTrialsBatch
        else:
            ## -- Split by Maximum number of events and time steps
            # - Handle None for nMaxNumTimeSteps
            nMaxNumTimeSteps = (
                nNumTimeSteps
                if self.nMaxNumTimeSteps is None
                else self.nMaxNumTimeSteps
            )
            # - Time step at which current batch starts
            nTSStartBatch = self._nTimeStep
            # - Corresponding index wrt vnTSInputEvents
            iStartBatch = 0
            # - Time step after evolution ends
            nTSEndEvo = nTSStartBatch + nNumTimeSteps
            while nTSStartBatch < nTSEndEvo:
                # - Endpoint of current batch
                nTSEndBatch = min(nTSStartBatch + nMaxNumTimeSteps, nTSEndEvo)
                # - Corresponding intex wrt vnTSInputEvents
                iEndBatch = np.searchsorted(vnTSInputEvents, nTSEndBatch)
                # - Correct if too many events are included
                if iEndBatch - iStartBatch > self.nMaxEventsPerBatch:
                    iEndBatch = iStartBatch + self.nMaxEventsPerBatch
                    nTSEndBatch = vnTSInputEvents[iEndBatch]
                # - Event data to be sent to FPGA
                vnTSInputEventsBatch = vnTSInputEvents[iStartBatch:iEndBatch]
                vnInputChannelsBatch = vnInputChannels[iStartBatch:iEndBatch]
                nNumTimeStepsBatch = nTSEndBatch - nTSStartBatch  # This is not the same as vnTSInputEventsBatch.size as the latter only contains events and not the complete time base
                if bVerbose:
                    nNumEventsBatch = iEndBatch - iStartBatch
                    print(
                        "Layer `{}`: Current batch input: {} s ({} timesteps)".format(
                            self.strName,
                            nNumTimeStepsBatch * self.tDt,
                            nNumTimeStepsBatch,
                        )
                        + ", {} events, starting with at {} s of {} s".format(
                            nNumEventsBatch,
                            nTSEndBatch * self.tDt,
                            nNumTimeSteps * self.tDt,
                        )
                    )
                yield (
                    vnTSInputEventsBatch,
                    vnInputChannelsBatch,
                    nTSStartBatch,
                    nNumTimeStepsBatch * self.tDt,
                )
                nTSStartBatch = nTSEndBatch
                iStartBatch = iEndBatch

    def evolve(
        self,
        tsInput: Optional[TSEvent] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = True,
    ) -> TSEvent:
        """
        evolve - Evolve the layer by queueing spikes, stimulating and recording

        :param tsInput:         TSEvent input time series, containing `self.nSize` channels
        :param tDuration:       float   Desired evolution duration, in seconds
        :param nNumTimeSteps:   int     Desired evolution duration, in integer steps of `self.tDt`
        :param bVerbose:        bool    Output information on evolution progress

        :return:                TSEvent spikes emitted by the neurons in this layer, during the evolution time
        """
        # - Compute duration for evolution
        if nNumTimeSteps is None:
            # - Determine nNumTimeSteps
            if tDuration is None:
                # - Determine tDuration
                assert (
                    tsInput is not None
                ), "Layer `{}`: One of `nNumTimeSteps`, `tsInput` or `tDuration` must be supplied".format(
                    self.strName
                )

                if tsInput.bPeriodic:
                    # - Use duration of periodic TimeSeries, if possible
                    tDuration = tsInput.tDuration

                else:
                    # - Evolve until the end of the input TImeSeries
                    tDuration = tsInput.tStop - self.t
                    assert tDuration > 0, (
                        "Layer `{}`: Cannot determine an appropriate evolution duration.".format(
                            self.strName
                        )
                        + " `tsInput` finishes before the current evolution time."
                    )
            nNumTimeSteps = int(np.floor((tDuration + ABS_TOLERANCE) / self.tDt))
        else:
            assert isinstance(
                nNumTimeSteps, int
            ), "Layer `{}`: nNumTimeSteps must be of type int.".format(self.strName)
            tDuration = nNumTimeSteps * self.tDt

        # - Lists for storing recorded data
        lTimeTrace = []
        lChannels = []

        # - Generator that splits inupt into batches
        gInputGenerator = self._batch_input_data(
            # - Clip tsInput to required duration
            tsInput.clip([self.t, self.t + tDuration]), nNumTimeSteps, bVerbose
        )
        
        # - Iterate over input batches
        for (
            vnTSInputEventsBatch,
            vnInputChannelsBatch,
            nTSStartBatch,
            tDurBatch,
        ) in gInputGenerator:
            vtTimeTraceBatch, vnChannelsBatch = self.controller.send_arrays(
                vnTimeSteps=vnTSInputEventsBatch - nTSStartBatch,
                vnChannels=vnInputChannelsBatch,
                tRecord=tDurBatch,
                vnNeuronIDs=self.vnVirtualNeuronIDs,
                vnRecordNeuronIDs=self.vnHWNeuronIDs,
                nTargetCoreMask=self._nInputCoreMask,
                nTargetChipID=self._nInputChipID,
                bPeriodic=False,
                bRecord=True,
                bTSEvent=False,
            )
            
            lChannels.append(vnChannelsBatch)
            lTimeTrace.append(vtTimeTraceBatch + nTSStartBatch * self.tDt)
            if bVerbose:
                print("Layer `{}`: Received event data".format(self.strName))

        # - Flatten out lTimeTrace and lChannels
        lTimeTrace = [t for vThisTrace in lTimeTrace for t in vThisTrace]
        lChannels = [ch for vTheseChannels in lChannels for ch in vTheseChannels]
        
        # - Convert recorded events into TSEvent object
        tsResponse = TSEvent(
            lTimeTrace, lChannels, nNumChannels=self.nSize, strName="DynapSE spikes"
        )

        # - Set layer time
        self._nTimeStep += nNumTimeSteps

        if bVerbose:
            print("Layer `{}`: Evolution successful.".format(self.strName))

        return tsResponse

    def _compile_weights_and_configure(self):
        """
        _compile_weights_and_configure - Configure DynapSE weights from the weight matrices
        """

        # - Get input to layer connections
        liPreSynE, liPostSynE, liPreSynI, liPostSynI = connectivity_matrix_to_prepost_lists(
            self.mfWIn
        )

        # - Connect input to layer

        # - Set excitatory input connections
        self.controller.dcNeuronConnector.add_connection_from_list(
            self._vVirtualNeurons[liPreSynE],
            self._vShadowNeurons[liPostSynE],
            [self.controller.synFE],
        )
        print(
            "Layer `{}`: Excitatory connections from virtual neurons to layer neurons have been set.".format(
                self.strName
            )
        )
        # - Set inhibitory input connections
        self.controller.dcNeuronConnector.add_connection_from_list(
            self._vVirtualNeurons[liPreSynI],
            self._vShadowNeurons[liPostSynI],
            [self.controller.synFI],
        )
        print(
            "Layer `{}`: Inhibitory connections from virtual neurons to layer neurons have been set.".format(
                self.strName
            )
        )

        # - Infer how many input neurons there are (i.e. any neuron that receives input from a virtual neuron)
        nNumInputNeurons = max(np.amax(liPostSynE), np.amax(liPostSynI))

        ## -- Set connections wihtin hardware layer
        liPreSynE, liPostSynE, liPreSynI, liPostSynI = connectivity_matrix_to_prepost_lists(
            self.mfWRec
        )

        viPreSynE = np.array(liPreSynE)
        viPreSynI = np.array(liPreSynI)
        viPostSynE = np.array(liPostSynE)
        viPostSynI = np.array(liPostSynI)

        # - Connections from input neurons to exceitatory neurons
        # Excitatory
        self.controller.dcNeuronConnector.add_connection_from_list(
            self._vShadowNeurons[viPreSynE[viPreSynE < nNumInputNeurons]],
            self._vShadowNeurons[viPostSynE[viPreSynE < nNumInputNeurons]],
            [self.controller.synFE],
        )
        print(
            "Layer `{}`: Excitatory connections from input neurons to reservoir neurons have been set.".format(
                self.strName
            )
        )
        # Inhibitory
        self.controller.dcNeuronConnector.add_connection_from_list(
            self._vShadowNeurons[viPreSynI[viPreSynI < nNumInputNeurons]],
            self._vShadowNeurons[viPostSynI[viPreSynI < nNumInputNeurons]],
            [self.controller.synFI],
        )
        print(
            "Layer `{}`: Inhibitory connections from input neurons to reservoir neurons have been set.".format(
                self.strName
            )
        )

        # - Connections between recurrently connected excitatory neurons and inhibitory neurons
        # Excitatory
        self.controller.dcNeuronConnector.add_connection_from_list(
            self._vShadowNeurons[viPreSynE[viPreSynE >= nNumInputNeurons]],
            self._vShadowNeurons[viPostSynE[viPreSynE >= nNumInputNeurons]],
            [self.controller.synSE],
        )
        print(
            "Layer `{}`: Excitatory connections within layer have been set.".format(
                self.strName
            )
        )
        # - Set inhibitory connections
        self.controller.dcNeuronConnector.add_connection_from_list(
            self._vShadowNeurons[viPreSynI[viPreSynI >= nNumInputNeurons]],
            self._vShadowNeurons[viPostSynI[viPreSynI >= nNumInputNeurons]],
            [self.controller.synFI],
        )
        print(
            "Layer `{}`: Inhibitory connections within layer have been set.".format(
                self.strName
            )
        )

        self.controller.model.apply_diff_state()
        print("Layer `{}`: Connections have been written to the chip.".format(self.strName))

    @property
    def cInput(self):
        return TSEvent

    @property
    def cOutput(self):
        return TSEvent

    @property
    def mfWIn(self):
        return self._mfWIn

    @mfWIn.setter
    def mfWIn(self, mfNewW):
        self._mfWIn = np.round(
            self._expand_to_shape(
                mfNewW, (self.nSizeIn, self.nSize), "mfWIn", bAllowNone=False
            )
        ).astype(int)

    @property
    def mfWRec(self):
        return self._mfWRec

    @mfWRec.setter
    def mfWRec(self, mfNewW):
        self._mfWRec = np.round(
            self._expand_to_shape(
                mfNewW, (self.nSize, self.nSize), "mfWRec", bAllowNone=False
            )
        ).astype(int)

    # mfW as alias for mfWRec
    @property
    def mfW(self):
        return self.mfWRec

    @mfW.setter
    def mfW(self, mfNewW):
        self.mfWRec = mfNewW

    # _mfW as alias for _mfWRec
    @property
    def _mfW(self):
        return self._mfWRec

    @_mfW.setter
    def _mfW(self, mfNewW):
        self._mfWRec = mfNewW

    @property
    def nMaxTrialsPerBatch(self):
        return self._nMaxTrialsPerBatch

    @nMaxTrialsPerBatch.setter
    def nMaxTrialsPerBatch(self, nNewMax):
        assert nNewMax is None or (
            type(nNewMax) == int and 0 < nNewMax
        ), "Layer `{}`: nMaxTrialsPerBatch must be an integer greater than 0 or None.".format(
            self.strName
        )
        self._nMaxTrialsPerBatch = nNewMax

    @property
    def nMaxNumTimeSteps(self):
        return self._nMaxNumTimeSteps

    @nMaxNumTimeSteps.setter
    def nMaxNumTimeSteps(self, nNewMax):
        assert nNewMax is None or (
            type(nNewMax) == int and 0 < nNewMax
        ), "Layer `{}`: nMaxNumTimeSteps must be an integer greater than 0 or None.".format(
            self.strName
        )
        if nNewMax > self.controller.nFpgaEventLimit * self.controller:
            warn("Layer `{}`: nMaxNumTimeSteps is larger than nFpgaEventLimit * nFpgaIsiLimit ({}).".format(
                self.strName, self.controller.nFpgaEventLimit * self.controller
            ))
        self._nMaxNumTimeSteps = nNewMax

    @property
    def tMaxBatchDur(self):
        return (
            None
            if self._nMaxNumTimeSteps is None
            else self._nMaxNumTimeSteps * self.tDt
        )

    @tMaxBatchDur.setter
    def tMaxBatchDur(self, tNewMax):
        assert tNewMax is None or (
            type(tNewMax) == int and 0 < tNewMax
        ), "Layer `{}`: tMaxBatchDur must be an integer greater than 0 or None.".format(
            self.strName
        )
        self._nMaxNumTimeSteps = (
            None if tNewMax is None else int(np.round(tNewMax / self.tDt))
        )

    @property
    def nMaxEventsPerBatch(self):
        return self._nMaxEventsPerBatch

    @nMaxEventsPerBatch.setter
    def nMaxEventsPerBatch(self, nNewMax):
        assert (
            type(nNewMax) == int and 0 < nNewMax <= self.controller.nFpgaEventLimit
        ), "Layer `{}`: nMaxEventsPerBatch must be an integer between 0 and {}.".format(
            self.strName, self.controller.nFpgaEventLimit
        )
        self._nMaxEventsPerBatch = nNewMax

    @property
    def vnVirtualNeuronIDs(self):
        return self._vnVirtualNeuronIDs

    @property
    def vnHWNeuronIDs(self):
        return self._vnHWNeuronIDs
    