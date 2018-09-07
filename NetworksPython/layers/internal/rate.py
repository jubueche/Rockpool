import numpy as np
from typing import Callable
from numba import njit

from ...timeseries import TimeSeries
from ..layer import Layer
from typing import Optional, Union, Tuple, List

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Relative tolerance for float comparions
fTolerance = 1e-5


### --- Helper functions


def isMultiple(a: float, b: float, fTolerance: float = fTolerance) -> bool:
    """
    isMultiple - Check whether a%b is 0 within some tolerance.
    :param a: float The number that may be multiple of b
    :param b: float The number a may be a multiple of
    :param fTolerance: float Relative tolerance
    :return bool: True if a is a multiple of b within some tolerance
    """
    fMinRemainder = min(a % b, b - a % b)
    return fMinRemainder < fTolerance * b


def print_progress(iCurr: int, nTotal: int, tPassed: float):
    print(
        "Progress: [{:6.1%}]    in {:6.1f} s. Remaining:   {:6.1f}".format(
            iCurr / nTotal, tPassed, tPassed * (nTotal - iCurr) / max(0.1, iCurr)
        ),
        end="\r",
    )


@njit
def fhReLu(vfX: np.ndarray) -> np.ndarray:
    vCopy = np.copy(vfX)
    vCopy[np.where(vfX < 0)] = 0
    return vCopy


@njit
def noisy(mX: np.ndarray, fStdDev: float) -> np.ndarray:
    """
    noisy - Add randomly distributed noise to each element of mX
    :param mX:  Array-like with values that noise is added to
    :param fStdDev: Float, the standard deviation of the noise to be added
    :return:        Array-like, mX with noise added
    """
    return fStdDev * np.random.randn(*mX.shape) + mX


### --- Functions used in connection with FFRateEuler class


@njit
def fhReLU(vfX: np.ndarray) -> np.ndarray:
    """
    Activation function for rectified linear units.
    :param vfX:             ndarray with current neuron potentials
    :return:                np.clip(vfX, 0, None)
    """
    mfCopy = np.copy(vfX)
    mfCopy[np.where(vfX < 0)] = 0
    return mfCopy


def get_ff_evolution_function(fhActivation: Callable[[np.ndarray], np.ndarray]):
    """
    get_ff_evolution_function: Construct a compiled Euler solver for a given activation function

    :param fhActivation: Callable (x) -> f(x)
    :return: Compiled function evolve_Euler_complete(vState, mfInput, mfW, nSize, nNumSteps, vfGain, vfBias, vfAlpha, fNoiseStd)
    """

    # - Compile an Euler solver for the desired activation function
    @njit
    def evolve_Euler_complete(
        vState: np.ndarray,
        mfInput: np.ndarray,
        mfW: np.ndarray,
        nSize: int,
        nNumSteps: int,
        vfGain: np.ndarray,
        vfBias: np.ndarray,
        vfAlpha: np.ndarray,
        fNoiseStd,
    ) -> np.ndarray:

        # - Initialise storage of layer output
        mfWeightedInput = mfInput @ mfW
        mfActivities = np.zeros((nNumSteps + 1, nSize))

        # - Loop over time steps. The updated vState already corresponds to
        # subsequent time step. Therefore skip state update in final step
        # and only update activation.
        for nStep in range(nNumSteps):
            # - Store layer activity
            mfActivities[nStep, :] = fhActivation(vState + vfBias)

            # - Evolve layer state
            vDState = -vState + noisy(vfGain * mfWeightedInput[nStep, :], fNoiseStd)
            vState += vDState * vfAlpha

        # - Compute final activity
        mfActivities[-1, :] = fhActivation(vState + vfBias)

        return mfActivities

    # - Return the compiled function
    return evolve_Euler_complete


def get_rec_evolution_function(fhActivation: Callable[[np.ndarray], np.ndarray]):
   """
   get_rec_evolution_function: Construct a compiled Euler solver for a given activation function

   :param fhActivation: Callable (x) -> f(x)
   :return: Compiled function evolve_Euler_complete(vState, nSize, mfW, mfInputStep, tDt, nNumSteps, vfBias, vtTau)
   """

   # - Compile an Euler solver for the desired activation function
   @njit
   def evolve_Euler_complete(
       vState: np.ndarray,
       nSize: int,
       mfW: np.ndarray,
       mfInputStep: np.ndarray,
       nNumSteps: int,
       tDt: float,
       vfBias: np.ndarray,
       vtTau: np.ndarray,
   ) -> np.ndarray:
       # - Initialise storage of network output
       mfActivity = np.zeros((nNumSteps + 1, nSize))

       # - Precompute tDt / vtTau
       vfLambda = tDt / vtTau

       # - Loop over time steps
       for nStep in range(nNumSteps):
           # - Evolve network state
           vfThisAct = fhActivation(vState + vfBias)
           vDState = -vState + mfInputStep[nStep, :] + mfW @ vfThisAct
           vState += vDState * vfLambda

           # - Store network state
           mfActivity[nStep, :] = vfThisAct

       # - Get final activation
       mfActivity[-1, :] = fhActivation(vState + vfBias)

       return mfActivity

   # - Return the compiled function
   return evolve_Euler_complete


### --- FFRateEuler class


class FFRateEuler(Layer):
    """ Feedforward layer consisting of rate-based neurons """

    def __init__(
        self,
        mfW: np.ndarray,
        tDt: float = None,
        strName: str = None,
        fNoiseStd: float = 0.,
        fhActivation: Callable[[np.ndarray], np.ndarray] = fhReLU,
        vtTau: Union[float, np.ndarray] = 10.,
        vfGain: Union[float, np.ndarray] = 1.,
        vfBias: Union[float, np.ndarray] = 0.,
    ):
        """
        FFRateEuler - Implement a feed-forward non-spiking neuron layer, with an Euler method solver

        :param mfW:             np.ndarray [MxN] Weight matrix
        :param tDt:             float Time step for Euler solver, in seconds
        :param strName:         string Name of this layer
        :param fNoiseStd:       float Noise std. dev. per second
        :param fhActivation:    Callable a = f(x) Neuron activation function (Default: ReLU)
        :param vtTau:           np.ndarray [Nx1] Vector of neuron time constants
        :param vfGain:          np.ndarray [Nx1] Vector of gain factors
        :param vfBias:          np.ndarray [Nx1] Vector of bias currents
        """

        # - Make sure some required parameters are set
        assert mfW is not None, \
            '`mfW` is required'

        assert vtTau is not None, \
            '`vtTau` is required'

        assert vfBias is not None, \
            '`vfBias` is required'

        assert vfGain is not None, \
            '`vfGain` is required'

        # - Set a reasonable tDt
        if tDt is None:
            tMinTau = np.min(vtTau)
            tDt = tMinTau / 10

        # - Call super-class initialiser
        super().__init__(
            mfW=np.asarray(mfW, float), tDt=tDt, fNoiseStd=fNoiseStd, strName=strName
        )

        # - Check all parameter shapes
        try:
            self.vtTau, self.vfGain, self.vfBias = map(
                self._correct_param_shape, (vtTau, vfGain, vfBias)
            )
        except AssertionError:
            raise AssertionError(
                "Numbers of elements in vtTau, vfGain and vfBias"
                + " must be 1 or match layer size."
            )

        # - Reset this layer state and set attributes
        self.reset_all()
        self.vfAlpha = self._tDt / self.vtTau
        self.fhActivation = fhActivation

    def _correct_param_shape(self, v) -> np.ndarray:
        """
        _correct_param_shape - Convert v to 1D-np.ndarray and verify
                              that dimensions match self.nSize
        :param v:   Float or array-like that is to be converted
        :return:    v as 1D-np.ndarray
        """
        v = np.array(v, dtype=float).flatten()
        assert v.shape in (
            (1,),
            (self.nSize,),
            (1, self.nSize),
            (self.nSize),
            1,
        ), "Numbers of elements in v must be 1 or match layer size"
        return v

    def evolve(
        self,
        tsInput: Optional[TimeSeries] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TimeSeries:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TimeSeries  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TimeSeries  output spike series

        """

        # - Prepare time base
        vtTimeBase, mfInput, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        mSamplesAct = self._evolveEuler(
            vState=self._vState,  # self._vState is automatically updated
            mfInput=mfInput,
            mfW=self._mfW,
            nSize=self._nSize,
            nNumSteps=nNumTimeSteps,
            vfGain=self._vfGain,
            vfBias=self._vfBias,
            vfAlpha=self._vfAlpha,
            # Without correction, standard deviation after some time will be
            # self._fNoiseStd * sqrt(self._vfAlpha/2)
            fNoiseStd=self._fNoiseStd * np.sqrt(2. / self._vfAlpha),
        )

        # - Increment internal time representation
        self._nTimeStep += nNumTimeSteps

        return TimeSeries(vtTimeBase, mSamplesAct)

    def stream(
        self, tDuration: float, tDt: float, bVerbose: bool = False
    ) -> Tuple[float, List[float]]:
        """
        stream - Stream data through this layer
        :param tDuration:   float Total duration for which to handle streaming
        :param tDt:         float Streaming time step
        :param bVerbose:    bool Display feedback

        :yield: (t, vState)

        :return: Final (t, vState)
        """

        # - Initialise simulation, determine how many tDt to evolve for
        if bVerbose:
            print("Layer: I'm preparing")
        vtTimeTrace = np.arange(0, tDuration + tDt, tDt)
        nNumSteps = np.size(vtTimeTrace) - 1
        nEulerStepsPerDt = int(np.round(tDt / self._tDt))

        if bVerbose:
            print("Layer: Prepared")

        # - Loop over tDt steps
        for nStep in range(nNumSteps):
            if bVerbose:
                print("Layer: Yielding from internal state.")
            if bVerbose:
                print("Layer: step", nStep)
            if bVerbose:
                print("Layer: Waiting for input...")

            # - Yield current output, receive input for next time step
            tupInput = (
                yield self._t,
                np.reshape(self._fhActivation(self._vState + self._vfBias), (1, -1)),
            )

            # - Set zero input if no input provided
            if tupInput is None:
                mfInput = np.zeros(nEulerStepsPerDt, self._nSizeIn)
            else:
                mfInput = np.repeat(
                    np.atleast_2d(tupInput[1][0, :]), nEulerStepsPerDt, axis=0
                )

            if bVerbose:
                print("Layer: Input was: ", tupInput)

            # - Evolve layer
            _ = self._evolveEuler(
                vState=self._vState,  # self._vState is automatically updated
                mfInput=mfInput,
                mfW=self._mfW,
                nSize=self._nSize,
                nNumSteps=nEulerStepsPerDt,
                vfGain=self._vfGain,
                vfBias=self._vfBias,
                vfAlpha=self._vfAlpha,
                # Without correction, standard deviation after some time will be
                # self._fNoiseStd * sqrt(self._vfAlpha/2)
                fNoiseStd=self._fNoiseStd * np.sqrt(2. / self._vfAlpha),
            )

            # - Increment time
            self._nTimeStep += nEulerStepsPerDt

        # - Return final state
        return (
            self.t,
            np.reshape(self._fhActivation(self._vState + self._vfBias), (1, -1)),
        )

    def train_rr(
        self,
        tsTarget: TimeSeries,
        tsInput: TimeSeries = None,
        fRegularize=0,
        bFirst=True,
        bFinal=False,
    ):
        """
        train_rr - Train self with ridge regression over one of possibly
                   many batches. Use Kahan summation to reduce rounding
                   errors when adding data to existing matrices from
                   previous batches.
        :param tsTarget:    TimeSeries - target for current batch
        :param tsInput:     TimeSeries - input to self for current batch
        :fRegularize:       float - regularization for ridge regression
        :bFirst:            bool - True if current batch is the first in training
        :bFinal:            bool - True if current batch is the last in training
        """

        # - Discrete time steps for evaluating input and target time series
        nNumTimeSteps = int(np.round(tsTarget.tDuration / self.tDt))
        vtTimeBase = self._gen_time_trace(tsTarget.tStart, nNumTimeSteps)

        if not bFinal:
            # - Discard last sample to avoid counting time points twice
            vtTimeBase = vtTimeBase[:-1]

        # - Make sure vtTimeBase does not exceed tsTarget
        vtTimeBase = vtTimeBase[vtTimeBase <= tsTarget.tStop]

        # - Prepare target data
        mfTarget = tsTarget(vtTimeBase)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            mfTarget
        ).any(), "nan values have been found in mfTarget (where: {})".format(
            np.where(np.isnan(mfTarget))
        )

        # - Check target dimensions
        if mfTarget.ndim == 1 and self.nSize == 1:
            mfTarget = mfTarget.reshape(-1, 1)

        assert (
            mfTarget.shape[-1] == self.nSize
        ), "Target dimensions ({}) does not match layer size ({})".format(
            mfTarget.shape[-1], self.nSize
        )

        # - Prepare input data

        # Empty input array with additional dimension for training biases
        mfInput = np.zeros((np.size(vtTimeBase), self.nSizeIn + 1))
        mfInput[:, -1] = 1

        if tsInput is not None:
            # Warn if intput time range does not cover whole target time range
            if (
                not tsInput.contains(vtTimeBase)
                and not tsInput.bPeriodic
                and not tsTarget.bPeriodic
            ):
                print(
                    "WARNING: tsInput (t = {} to {}) does not cover ".format(
                        tsInput.tStart, tsInput.tStop
                    )
                    + "full time range of tsTarget (t = {} to {})\n".format(
                        tsTarget.tStart, tsTarget.tStop
                    )
                    + "Assuming input to be 0 outside of defined range.\n"
                )

            # - Sample input trace and check for correct dimensions
            mfInput[:, :-1] = self._check_input_dims(tsInput(vtTimeBase))
            # - Treat "NaN" as zero inputs
            mfInput[np.where(np.isnan(mfInput))] = 0

        else:
            print(
                "No tsInput defined, assuming input to be 0 and only training biases."
            )

        # - For first batch, initialize summands
        if bFirst:
            # Matrices to be updated for each batch
            self.mfXTY = np.zeros(
                (self.nSizeIn + 1, self.nSize)
            )  # mfInput.T (dot) mfTarget
            self.mfXTX = np.zeros(
                (self.nSizeIn + 1, self.nSizeIn + 1)
            )  # mfInput.T (dot) mfInput
            # Corresponding Kahan compensations
            self.mfKahanCompXTY = np.zeros_like(self.mfXTY)
            self.mfKahanCompXTX = np.zeros_like(self.mfXTX)

        # - New data to be added, including compensation from last batch
        #   (Matrix summation always runs over time)
        mfUpdXTY = mfInput.T @ mfTarget - self.mfKahanCompXTY
        mfUpdXTX = mfInput.T @ mfInput - self.mfKahanCompXTX

        if not bFinal:
            # - Update matrices with new data
            mfNewXTY = self.mfXTY + mfUpdXTY
            mfNewXTX = self.mfXTX + mfUpdXTX
            # - Calculate rounding error for compensation in next batch
            self.mfKahanCompXTY = (mfNewXTY - self.mfXTY) - mfUpdXTY
            self.mfKahanCompXTX = (mfNewXTX - self.mfXTX) - mfUpdXTX
            # - Store updated matrices
            self.mfXTY = mfNewXTY
            self.mfXTX = mfNewXTX

        else:
            # - In final step do not calculate rounding error but update matrices directly
            self.mfXTY += mfUpdXTY
            self.mfXTX += mfUpdXTX

            # - Weight and bias update by ridge regression
            mfSolution = np.linalg.solve(
                self.mfXTX + fRegularize * np.eye(self.nSizeIn + 1), self.mfXTY
            )
            self.mfW = mfSolution[:-1, :]
            self.vfBias = mfSolution[-1, :]

            # - Remove dat stored during this trainig
            self.mfXTY = self.mfXTX = self.mfKahanCompXTY = self.mfKahanCompXTX = None

    # @njit
    # def potential(self, vInput: np.ndarray) -> np.ndarray:
    #     return (self._vfAlpha * noisy(vInput@self.mfW*self.vfGain + self.vfBias, self.fNoiseStd)
    #             + (1-self._vfAlpha)*self.vState)

    def __repr__(self):
        return "FFRateEuler layer object `{}`.\nnSize: {}, nSizeIn: {}   ".format(
            self.strName, self.nSize, self.nSizeIn
        )

    @property
    def vActivation(self):
        return self.fhActivation(self.vState)

    ### --- properties

    @property
    def vtTau(self):
        return self._vtTau

    @vtTau.setter
    def vtTau(self, vNewTau):
        vNewTau = self._correct_param_shape(vNewTau)
        if not (vNewTau >= self._tDt).all():
            raise ValueError("All vtTau must be at least tDt.")
        self._vtTau = vNewTau
        self._vfAlpha = self._tDt / vNewTau

    @property
    def vfAlpha(self):
        return self._vfAlpha

    @vfAlpha.setter
    def vfAlpha(self, vNewAlpha):
        vNewAlpha = self._correct_param_shape(vNewAlpha)
        if not (vNewAlpha <= 1).all():
            raise ValueError("All vfAlpha must be at most 1.")
        self._vfAlpha = vNewAlpha
        self._vtTau = self._tDt / vNewAlpha

    @property
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vNewBias):
        self._vfBias = self._correct_param_shape(vNewBias)

    @property
    def vfGain(self):
        return self._vfGain

    @vfGain.setter
    def vfGain(self, vNewGain):
        self._vfGain = self._correct_param_shape(vNewGain)

    @property
    def fhActivation(self):
        return self._fhActivation

    @fhActivation.setter
    def fhActivation(self, f):
        self._fhActivation = f
        self._evolveEuler = get_ff_evolution_function(f)

    @Layer.tDt.setter
    def tDt(self, tNewDt):
        if not (self.vtTau >= tNewDt).all():
            raise ValueError("All vtTau must be at least tDt.")
        self._tDt = tNewDt
        self._vfAlpha = tNewDt / self._vtTau


### --- PassThrough Class


class PassThrough(FFRateEuler):
    """ Neuron states directly correspond to input, but can be delayed. """

    def __init__(
        self,
        mfW: np.ndarray,
        tDt: float = 1.,
        fNoiseStd: float = 0.,
        vfBias: Union[float, np.ndarray] = 0.,
        tDelay: float = 0.,
        strName: str = None,
    ):
        """
        PassThrough - Implement a feed-forward layer that simply passes input (possibly delayed)

        :param mfW:         np.ndarray [MxN] Weight matrix
        :param tDt:         float Time step for Euler solver, in seconds
        :param fNoiseStd:   float Noise std. dev. per second
        :param vfBias:      np.ndarray [Nx1] Vector of bias currents
        :param tDelay:      float Delay between input and output, in seconds
        :param strName:     string Name of this layer
        """
        # - Set delay
        self._nDelaySteps = 0 if tDelay is None else int(np.round(tDelay / tDt))

        # - Call super-class initialiser
        super().__init__(
            mfW=mfW,
            tDt=tDt,
            fNoiseStd=fNoiseStd,
            fhActivation=lambda x: x,
            vfBias=vfBias,
            strName=strName,
        )

        self.reset_all()

    def reset_buffer(self):
        if self.tDelay != 0:
            vtBuffer = np.arange(self._nDelaySteps + 1) * self._tDt
            self.tsBuffer = TimeSeries(vtBuffer, np.zeros((len(vtBuffer), self.nSize)))
        else:
            self.tsBuffer = None

    def evolve(
        self,
        tsInput: Optional[TimeSeries] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TimeSeries:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TimeSeries  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TimeSeries  output spike series

        """

        # - Prepare time base
        vtTimeBase, mfInput, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        # - Apply input weights and add noise
        mfInProcessed = noisy(mfInput @ self.mfW, self.fNoiseStd)

        if self.tsBuffer is not None:
            # - Combined time trace for buffer and processed input
            nNumTimeStepsComb = nNumTimeSteps + self._nDelaySteps
            vtTimeComb = self._gen_time_trace(self.t, nNumTimeStepsComb)

            # - Array for buffered and new data
            mfSamplesComb = np.zeros((vtTimeComb.size, self.nSizeIn))
            nStepsIn = vtTimeBase.size

            # - Buffered data: last point of buffer data corresponds to self.t,
            #   which is also part of current input
            mfSamplesComb[:-nStepsIn] = self.tsBuffer.mfSamples[:-1]

            # - Processed input data (weights and noise)
            mfSamplesComb[-nStepsIn:] = mfInProcessed

            # - Output data
            mfSamplesOut = mfSamplesComb[:nStepsIn]

            # - Update buffer with new data
            self.tsBuffer.mfSamples = mfSamplesComb[nStepsIn - 1 :]

        else:
            # - Undelayed processed input
            mfSamplesOut = mfInProcessed

        # - Update state and time
        self.vState = mfSamplesOut[-1]
        self._nTimeStep += nNumTimeSteps

        # - Return time series with output data and bias
        return TimeSeries(vtTimeBase, mfSamplesOut + self.vfBias)

    def __repr__(self):
        return "PassThrough layer object `{}`.\nnSize: {}, nSizeIn: {}, tDelay: {}".format(
            self.strName, self.nSize, self.nSizeIn, self.tDelay
        )

    def print_buffer(self, **kwargs):
        if self.tsBuffer is not None:
            self.tsBuffer.print(**kwargs)
        else:
            print("This layer does not use a delay.")

    @property
    def mfBuffer(self):
        if self.tsBuffer is not None:
            return self.tsBuffer.mfSamples
        else:
            print("This layer does not use a delay.")

    def reset_state(self):
        super().reset_state()
        self.reset_buffer()

    def reset_all(self):
        super().reset_all()
        self.reset_buffer()

    @property
    def tDelay(self):
        return self._nDelaySteps * self.tDt

    @property
    def nDelaySteps(self):
        return self._nDelaySteps

    # @tDelay.setter
    # def tDelay(self, tNewDelay):
    # Some method to extend self.tsBuffer


class RecRateEuler(Layer):
    def __init__(
        self,
        mfW: np.ndarray,
        vfBias: np.ndarray = 0,
        vtTau: np.ndarray = 1,
        fhActivation: Callable[[np.ndarray], np.ndarray] = fhReLu,
        tDt: float = None,
        fNoiseStd: float = 0,
        strName: str = None,
    ):
        """
        RecRate: Implement a recurrent layer with firing rate neurons

        :param mfW:             np.ndarray (NxN) matrix of recurrent weights
        :param vfBias:          np.ndarray (N) vector (or scalar) of bias currents
        :param vtTau:           np.ndarray (N) vector (or scalar) of neuron time constants
        :param fhActivation:    Callable (x) -> f(x) Activation function
        :param tDt:             float Time step for integration (Euler method)
        :param fNoiseStd:       float Std. Dev. of state noise injected at each time step
        :param strName:           str Name of this layer
        """

        # - Call super-class init
        super().__init__(mfW=mfW, strName=strName)

        # - Check size and shape of `mfW`
        assert len(mfW.shape) == 2, \
            '`mfW` must be a matrix with 2 dimensions'
        assert mfW.shape[0] == mfW.shape[1], \
            '`mfW` must be a square matrix'

        # - Assign properties
        self.vfBias = vfBias
        self.vtTau = vtTau
        self.fhActivation = fhActivation
        self.fNoiseStd = fNoiseStd

        if tDt is not None:
            self.tDt = tDt

        # - Reset the internal state
        self.reset_all()

    ### --- Properties

    @property
    def vfBias(self) -> np.ndarray:
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vfNewBias: np.ndarray):
        self._vfBias = self._expand_to_net_size(vfNewBias, "vfNewBias")

    @property
    def vtTau(self) -> np.ndarray:
        return self._vtTau

    @vtTau.setter
    def vtTau(self, vtNewTau: np.ndarray):
        self._vtTau = self._expand_to_net_size(vtNewTau, "vtNewTau")

        # - Ensure tDt is reasonable for numerical accuracy
        self.tDt = np.min(self.vtTau) / 10

    ### --- State evolution method

    def evolve(
        self,
        tsInput: Optional[TimeSeries] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TimeSeries:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TimeSeries  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:            TimeSeries  output spike series

        """

        # - Prepare time base
        vtTimeBase, mfInputStep, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        # - Generate a noise trace
        # Noise correction: Standard deviation after some time would be fNoiseStd * sqrt(0.5*tDt/vtTau)
        mfNoiseStep = (
            np.random.randn(np.size(vtTimeBase), self.nSize)
            * self.fNoiseStd
            * np.sqrt(2. * self._vtTau / self._tDt)
        )

        # - Call Euler method integrator
        #   Note: Bypass setter method for .vState
        mfActivity = self._evolveEuler(
            self._vState,
            self._nSize,
            self._mfW,
            mfInputStep + mfNoiseStep,
            nNumTimeSteps,
            self._tDt,
            self._vfBias,
            self._vtTau,
        )

        # - Increment internal time representation
        self._nTimeStep += nNumTimeSteps

        # - Construct a return TimeSeries
        return TimeSeries(vtTimeBase, mfActivity)

    def stream(
        self, tDuration: float, tDt: float, bVerbose: bool = False
    ) -> Tuple[float, List[float]]:
        """
        stream - Stream data through this layer
        :param tDuration:   float Total duration for which to handle streaming
        :param tDt:         float Streaming time step
        :param bVerbose:    bool Display feedback

        :yield: (t, vState)

        :return: Final (t, vState)
        """

        # - Initialise simulation, determine how many tDt to evolve for
        if bVerbose:
            print("Layer: I'm preparing")
        vtTimeTrace = np.arange(0, tDuration + tDt, tDt)
        nNumSteps = np.size(vtTimeTrace) - 1
        nEulerStepsPerDt = int(tDt / self._tDt)

        # - Generate a noise trace
        mfNoiseStep = (
            np.random.randn(np.size(vtTimeBase), self.nSize)
            * self.fNoiseStd
            * np.sqrt(2. * self._vtTau / self._tDt)
        )

        if bVerbose:
            print("Layer: Prepared")

        # - Loop over tDt steps
        for nStep in range(nNumSteps):
            if bVerbose:
                print("Layer: Yielding from internal state.")
            if bVerbose:
                print("Layer: step", nStep)
            if bVerbose:
                print("Layer: Waiting for input...")

            # - Yield current activity, receive input for next time step
            tupInput = (
                yield self._t,
                np.reshape(self._fhActivation(self._vState + self._vfBias), (1, -1)),
            )

            # - Set zero input if no input provided
            if tupInput is None:
                mfInput = np.zeros(nEulerStepsPerDt, self._nSizeIn)
            else:
                mfInput = np.repeat(
                    np.atleast_2d(tupInput[1][0, :]), nEulerStepsPerDt, axis=0
                )

            if bVerbose:
                print("Layer: Input was: ", tupInput)

            # - Evolve layer
            _ = self._evolveEuler(
                vState=self._vState,  # self._vState is automatically updated
                nSize=self._nSize,
                mfW=self._mfW,
                mfInputStep=mfInput + mfNoiseStep[nStep, :],
                nNumSteps=nEulerStepsPerDt,
                tDt=self._tDt,
                vfBias=self._vfBias,
                vtTau=self._vtTau,
            )

            # - Increment time
            self._nTimeStep += nEulerStepsPerDt

        # - Return final activity
        return (
            self.t,
            np.reshape(self._fhActivation(self._vState + self._vfBias), (1, -1)),
        )

    ### --- Properties

    @Layer.tDt.setter
    def tDt(self, tNewDt: float):
        # - Check that the time step is reasonable
        tMinTau = np.min(self.vtTau)
        assert tNewDt <= tMinTau / 10, "`tNewDt` must be <= {}".format(tMinTau / 10)

        # - Call super-class setter
        super(RecRateEuler, RecRateEuler).tDt.__set__(self, tNewDt)

    @property
    def fhActivation(self):
        return self._fhActivation

    @fhActivation.setter
    def fhActivation(self, fhNewActivation):
        self._fhActivation = fhNewActivation

        # - Build a state evolution function
        self._evolveEuler = get_rec_evolution_function(fhNewActivation)
