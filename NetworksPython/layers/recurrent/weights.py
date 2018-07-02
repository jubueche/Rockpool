from typing import Callable
import numpy as np
import scipy.stats as stats
from copy import deepcopy
from brian2.units.allunits import mvolt
import random


def RndmSparseEINet(nResSize : int,
                    fConnectivity : float = 1,
                    fhRand : Callable[[int], np.ndarray] = np.random.rand,
                    bPartitioned: bool = False,
                    fRatioExc: float = 0.5,
                    fScaleInh: float = 1,
                    fNormalize: float = 0.95) -> np.ndarray:
    """
    RndmSparseEINet - Return a (sparse) matrix defining reservoir weights
    
    :param nResSize:        int Number of reservoir units
    :param fConnectivity:   float Ratio of non-zero weight matrix elements
                                  (must be between 0 and 1)
    :param fhRand:          Function used to draw random weights. Must accept an integer n 
                            as argument and return n positive values.
    :param bPartitioned:    bool  Partition weight matrix into excitatory and inhibitory
    :param fRatioExc:       float Ratio of excitatory weights (must be between 0 and 1)
    :param fInhiblScale:    float Scale of negative weights to positive weights
    :param fNormalize:      float If not None, matrix is normalized so that
                                  its spectral radius equals fNormalize
    :return:                np.ndarray Weight matrix
    """ 

    # - Make sure parameters are in correct range
    fConnectivity = np.clip(fConnectivity, 0, 1)
    fRatioExc = np.clip(fRatioExc, 0, 1)

    # - Number of non-zero elements in matrix
    nNumWeights = int(fConnectivity*nResSize**2)
    
    # - Array for storing connection strengths
    mfWeights = np.zeros(nResSize**2)
    # - Draw non-zero weights
    mfWeights[:nNumWeights] = fhRand(nNumWeights)

    if not bPartitioned:
        # - Number inhibitory connections
        nNumInhWeights = int(nNumWeights * (1-fRatioExc))
        # - Multiply inhibitory connections with -fScaleInh
        mfWeights[:nNumInhWeights] *= -fScaleInh

    # - Shuffle weights and bring array into 2D-form
    np.random.shuffle(mfWeights)
    mfWeights = mfWeights.reshape(nResSize, nResSize)

    if bPartitioned:
        # - Number of excitatory neurons
        nNumExcNeurons = int(nResSize * fRatioExc)
        # - All rows with index > nNumExcNeurons correspond to inhibitory neurons
        #   Multiply coresponding elements with -1 and scale with fScaleInh
        mfWeights[nNumExcNeurons:] *= -fScaleInh

    if fNormalize is not None:
        # Normalize weights matrix so that its spectral radius = fNormalize
        vEigenvalues = np.linalg.eigvals(mfWeights)
        mfWeights *= fNormalize/np.amax(np.abs(vEigenvalues))

    return mfWeights

def RandomEINet(nNumExc: int,
                nNumInh: int,
                fInhWFactor: float = 1,
                fhRand: Callable[[int], float] = lambda n: np.random.randn(n, n) / np.sqrt(n)) -> np.ndarray:
    """
    RandomEINet - Generate a random nicely-tuned real-valued reservoir matrix
    :param nNumExc:         Number of excitatory neurons in the network
    :param nNumInh:         Number of inhibitory neurons in the network
    :param fInhWFactor:     Factor relating total inhibitory and excitatory weight (w_inh = fInhWFactor * w_exc) default: 1
    :param fhRand:          Function used to draw initial random weights. Default: numpy.random.randn

    :return:                Network connectivity weight matrix
    """

    # - Generate base connectivity matrix
    mfW = fhRand(nNumExc + nNumInh)

    # - Enforce excitatory and inhibitory partitioning
    vnExc = range(nNumExc)
    vnInh = [n + nNumExc for n in range(nNumInh)]

    mfWE = mfW[:, vnExc]
    mfWE[mfWE < 0] = 0
    mfWE /= np.sum(mfWE, 0)

    mfWI = mfW[:, vnInh]
    mfWI[mfWI > 0] = 0
    mfWI = mfWI / np.sum(mfWI, 0) * -np.abs(fInhWFactor)

    mfW = np.concatenate((mfWE, mfWI), 1)

    return mfW

def WilsonCowanNet(nNumNodes: int,
                   fSelfExc: float = 1,
                   fSelfInh: float = 1,
                   fExcSigma: float = 1,
                   fInhSigma: float = 1,
                   fhRand: Callable[[int], float] = lambda n: np.random.randn(n, n) / np.sqrt(n)) -> (np.ndarray, np.ndarray):
    """
    WilsonCowanNet - FUNCTION Define a Wilson-Cowan network of oscillators

    :param nNumNodes: Number of (E+I) nodes in the network
    :param fSelfExc:  Strength of self-excitation autapse. Default 1.0
    :param fSelfInh:  Strength of self-inhibition autapse. Default 1.0
    :param fExcSigma: Sigma of random excitatory connections. Default 1.0
    :param fInhSigma: Sigma of random inhibitory connections. Default 1.0
    :param fhRand:    Function used to draw random weights. Default: numpy.random.randn
    :return:          2N x 2N weight matrix mfW
    """

    # - Check arguments, enforce reasonable defaults

    # - Build random matrices
    mfEE = np.clip(fhRand(nNumNodes), 0, None) * fExcSigma + np.identity(nNumNodes) * fSelfExc
    mfIE = np.clip(fhRand(nNumNodes), 0, None) * fExcSigma + np.identity(nNumNodes) * fSelfExc
    mfEI = np.clip(fhRand(nNumNodes), None, 0) * fInhSigma + np.identity(nNumNodes) * -np.abs(fSelfInh)
    mfII = np.clip(fhRand(nNumNodes), None, 0) * fInhSigma + np.identity(nNumNodes) * -np.abs(fSelfInh)

    # - Normalise matrix components
    fENorm = fExcSigma * stats.norm.pdf(0, 0, fExcSigma) * nNumNodes + fSelfExc
    fINorm = fInhSigma * stats.norm.pdf(0, 0, fInhSigma) * nNumNodes + np.abs(fSelfInh)
    mfEE = mfEE / np.sum(mfEE, 0) * fENorm
    mfIE = mfIE / np.sum(mfIE, 0) * fENorm
    mfEI = mfEI / np.sum(mfEI, 0) * -fINorm
    mfII = mfII / np.sum(mfII, 0) * -fINorm

    # - Compose weight matrix
    mfW = np.concatenate((np.concatenate((mfEE, mfIE)),
                          np.concatenate((mfEI, mfII))), axis = 1)

    return mfW

def WipeNonSwitchingEigs(mfW: np.ndarray, vnInh: np.ndarray = None, fInhTauFactor: float = 1) -> np.ndarray:
    """
    WipeNonSwitchingEigs - Eliminate eigenvectors that do not lead to a partition switching

    :param mfW:             Network weight matrix [N x N]
    :param vnInh:           Vector [M x 1] of indices of inhibitory neurons. Default: None
    :param fInhTauFactor:   Factor relating inhibitory and excitatory time constants. tau_i = f * tau_e, tau_e = 1 Default: 1
    :return:                (mfW, mfJ) Weight matrix and estimated Jacobian
    """

    nResSize = mfW.shape[0]

    # - Compute Jacobian
    mfJ = mfW - np.identity(nResSize)

    if vnInh is not None:
        mfJ[vnInh, :] /= fInhTauFactor

    # - Numerically estimate eigenspectrum
    [vfD, mfV] = np.linalg.eig(mfJ)

    # - Identify and wipe non-switching eigenvectors
    mfNormVec = mfV / np.sign(vfD)
    vbNonSwitchingPartition = np.all(mfNormVec > 0, 0)
    vbNonSwitchingPartition[0] = False
    print('Number of eigs wiped: ' + str(np.sum(vbNonSwitchingPartition)))
    vfD[vbNonSwitchingPartition] = 0

    # - Reconstruct Jacobian and weight matrix
    mfJHat = np.real(mfV @ np.diag(vfD) @ np.linalg.inv(mfV))
    mfWHat = mfJHat

    if vnInh is not None:
        mfWHat[vnInh, :] *= fInhTauFactor

    mfWHat += np.identity(nResSize)

    # - Attempt to rescale weight matrix for optimal dynamics
    mfWHat *= fInhTauFactor * 30

    return mfWHat, mfJHat


def UnitLambdaNet(nResSize: int) -> np.ndarray:
    """
    UnitLambdaNet - Generate a network from Norm(0, sqrt(N))

    :param nResSize: Number of neurons in the network
    :return:         mfW: weight matrix
    """

    # - Draw from a Normal distribution
    mfW = np.random.randn(nResSize, nResSize) / np.sqrt(nResSize)
    return mfW

def DiscretiseWeightMatrix(mfW: np.ndarray, nMaxConnections: int = 3,
                           nLimitInputs: int = None, nLimitOutputs: int = None) \
        -> (np.ndarray, np.ndarray, float, float):
    """
    DiscretiseWeightMatrix - Discretise a weight matrix by strength

    :param mfW:             an arbitrary real-valued weight matrix.
    :param nMaxConnections: the integer maximum number of synaptic connections that may be made between
                            two neurons. Excitatory and inhibitory weights will be discretised
                            separately. Default: 3
    :param nLimitInputs:    integer Number of permissable inputs per neuron
    :param nLimitOutputs:   integer Number of permissable outputs per neuron
    :return: (mfWD, mnNumConns, fEUnitary, fIUnitary)
    """

    # - Make a copy of the input
    mfW = deepcopy(mfW)
    mfWE = mfW * (mfW > 0)
    mfWI = mfW * (mfW < 0)

    # - Select top N inputs per neuron
    if nLimitOutputs is not None:
        mfWE = np.array([row * np.array(np.argsort(-row) < np.round(nLimitOutputs/2), 'float') for row in mfWE.T]).T
        mfWI = np.array([row * np.array(np.argsort(-abs(row)) < np.round(nLimitOutputs/2), 'float') for row in
                         mfWI.T]).T

    if nLimitInputs is not None:
        mfWE = np.array([row * np.array(np.argsort(-row) < np.round(nLimitInputs/2), 'float') for row in mfWE])
        mfWI = np.array([row * np.array(np.argsort(-abs(row)) < np.round(nLimitInputs/2), 'float') for row in mfWI])

    # - Determine unitary connection strengths. Clip at max absolute values
    fEUnitary = np.max(mfW) / nMaxConnections
    fIUnitary = np.max(-mfW) / nMaxConnections

    # - Determine number of unitary connections
    if np.any(mfW > 0):
        mnEConns = np.round(mfW / fEUnitary) * (mfW > 0)
    else:
        mnEConns = 0

    if np.any(mfW < 0):
        mnIConns = -np.round(mfW / fIUnitary) * (mfW < 0)
    else:
        mnIConns = 0

    mnNumConns = mnEConns - mnIConns

    # - Discretise real-valued weight matrix
    mfWD = mnEConns * fEUnitary - mnIConns * fIUnitary

    # - Return matrices
    return mfWD, mnNumConns


def IAFSparseNet(nResSize: int = 100,
                 fMean: float = None, fStd: float = None,
                 fDensity: float = 1.0) -> np.ndarray:
    """
    IAFSparseNet - Return a random sparse reservoir, scaled for a standard IAF spiking network

    :param nResSize:    int Number of neurons in reservoir. Default: 100
    :param fMean:       float Mean connection strength. Default: -4.5mV
    :param fStd:        float Std. Dev. of connection strengths. Default: 5.5mV
    :param fDensity:    float 0..1 Density of connections. Default: 1 (full connectivity)
    :return:            [N x N] array Weight matrix
    """

    # - Check inputs
    assert (fDensity >= 0.0) and (fDensity <= 1.0), \
        '`fDensity` must be between 0 and 1.'

    # - Set default values
    if fMean is None:
        fMean = -4.5 * mvolt / fDensity / nResSize

    if fStd is None:
        fStd = 5.5 * mvolt / np.sqrt(fDensity)

    # - Determine sparsity
    nNumConnPerRow = int(fDensity * nResSize)
    nNumNonConnPerRow = nResSize - nNumConnPerRow
    mbConnection = [random.sample([1] * nNumConnPerRow + [0] * nNumNonConnPerRow, nResSize) for _ in
                    range(nResSize)]
    vbConnection = np.array(mbConnection).reshape(-1)

    # - Randomise recurrent weights
    mfW = (np.random.randn(nResSize ** 2) * (np.asarray(fStd) ** 2) + np.asarray(fMean)) * vbConnection
    return mfW.reshape(nResSize, nResSize)
