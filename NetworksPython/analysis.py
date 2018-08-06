import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import fftconvolve

from bokeh import plotting as bk


def analyze(
    vOutput,
    vTarget,
    vInput,
    fThrDetect,
    nWindow=200,
    nClose=100,
    bVerbose=True,
    bPlot=True,
    mfTarget=None,
):
    """Analyze output signal vOutput wrt to target:
        - Sensitivity: Which of the anomalies have been detected?
            Detected means that there is at least one detection during
              or up to nWindow time steps after the anomaly
            A detection is when vOutput crosses fThrDetect from below
              and there has been no detection within the last nClose
              time steps
            If two anomalies happen with less than
              nClose time steps between their beginning, detecting
              both is unlikely. Therefore in this case it is sufficient
              if vOutput is above threshold.
        - False detections: Each detection that happens within a normal
          interval counts as false detection.
            A normal interval is an interval where no anomaly has
              occured within the previous nWindow time steps.
        - For the specificity, each normal interval in which one or
          more detections happen counts as false positive
    """

    nDuration = len(vOutput)
    vbAboveThr = (np.abs(vOutput) > fThrDetect).flatten()

    # Detects
    viDetects = find_detects(vOutput.flatten(), fThrDetect, nClose)
    vbDetects = np.zeros(nDuration, bool)
    vbDetects[viDetects] = True

    # - Using multi-dimensional target allows for more precise analysis in case of overlapping anomalies
    if mfTarget is not None:
        # Anomaly starts and ends
        vnAnomStarts, vnAnomEnds = anom_starts_ends_multi(mfTarget)
        # Detected anomalies
        lbDetectedAnom = detected_anoms_multi(
            vOutput.reshape(-1, 1), mfTarget, vbDetects, fThrDetect, nWindow, nClose
        )

    else:
        # Anomaly starts and ends
        vnAnomStarts, vnAnomEnds = anom_starts_ends(vTarget)
        # Detected anomalies
        lbDetectedAnom = detected_anoms(
            vOutput, vTarget, vbDetects, fThrDetect, nWindow, nClose
        )

    # - Sensitivity
    nTruePos = np.sum(lbDetectedAnom)
    nFalseNeg = len(lbDetectedAnom) - nTruePos
    fSensitivity = nTruePos / len(lbDetectedAnom)

    # - List with number of false detects for each interval, starts and ends of normal intervals
    lnFalseDetects, viNormalStarts, viNormalEnds = detects_in_normal_multi(
        vTarget.reshape(-1, 1), vbDetects, nWindow
    )

    # Specificity
    nTrueNeg = np.sum(
        np.asarray(lnFalseDetects) == 0
    )  # Number of normal intervals with no detection
    nFalseDetects = np.sum(lnFalseDetects)
    nFalseDetectIntervals = len(lnFalseDetects) - nTrueNeg
    fSpecificity = nTrueNeg / len(
        lnFalseDetects
    )  # len(lnFalseDetects) corresponds to number of normal intervals

    nErrors = nFalseNeg + nFalseDetectIntervals

    v0Target = np.r_[np.zeros(nWindow), vTarget]  # vTarget with leading 0s
    vbAnomal = np.array(
        [(v0Target[i : i + nWindow + 1] >= 0.5).any() for i in range(len(vTarget))]
    )

    # Symptom-wise analysis
    dSymptoms = dict()
    if mfTarget is not None:
        for iSymptom in range(mfTarget.shape[1]):

            dSymptoms[iSymptom] = dict()
            vCurrTarg = mfTarget[:, iSymptom]

            # Detected anomalies
            lbDetectedSympt = detected_anoms(
                vOutput.flatten(), vCurrTarg, vbDetects, fThrDetect, nWindow, nClose
            )
            # Sensitivity
            dSymptoms[iSymptom]["nTruePos"] = np.sum(lbDetectedSympt)
            dSymptoms[iSymptom]["nFalseNeg"] = len(lbDetectedSympt) - np.sum(
                lbDetectedSympt
            )
            dSymptoms[iSymptom]["fSensitivity"] = (
                np.sum(lbDetectedSympt) / len(lbDetectedSympt) if lbDetectedSympt else 1
            )

    if bVerbose:
        print(
            "Sensitivity : {:.1%} ({} of {} anomalies detected)".format(
                fSensitivity, nTruePos, len(lbDetectedAnom)
            )
        )
        print(
            "Specificity : {:.1%} ({} of {} normal intervals correct)".format(
                fSpecificity, nTrueNeg, len(lnFalseDetects)
            )
        )
        print(
            "{} Errors: \n {} of {} anomalies missed,".format(
                nErrors, nFalseNeg, len(lbDetectedAnom)
            )
            + " {} false detections (in {} intervals)".format(
                nFalseDetects, nFalseDetectIntervals
            )
        )
    if bPlot:
        # - Plotting
        fig, ax = plt.subplots()
        ax.plot(vOutput)
        ax.plot(vTarget)
        ax.plot([0, nDuration], [fThrDetect, fThrDetect], "k--")
        ax.plot(0.2 * vInput, lw=2, color="k", alpha=0.4, zorder=-1)
        # Higlight missed anomalies red, detected anomalies green
        for i, (s, e) in enumerate(zip(vnAnomStarts, vnAnomEnds)):
            if lbDetectedAnom[i]:
                # ax.fill_between(np.arange(s, e+nWindow), -1, 1, color='g', alpha=0.2)
                ax.plot([s, e + nWindow], [1.5, 1.5], color="g", lw=15, alpha=0.5)
            else:
                # ax.fill_between(np.arange(s, e+nWindow), -1, 1, color='r', alpha=0.2)
                ax.plot([s, e + nWindow], [1.5, 1.5], color="r", lw=15, alpha=0.5)

        # Highlight false positive intervals red
        for i, (s, e) in enumerate(zip(viNormalStarts, viNormalEnds)):
            if lnFalseDetects[i] > 0.1:
                # ax.fill_between(np.arange(s, e), -1.5, -1, color='r', alpha=0.2)
                ax.plot([s, e], [-0.75, -0.75], color="r", lw=15, alpha=0.5)
        # Mark false detections with vertical red line
        # Indicate time points during and up to nWindow time steps after anomaly
        for i in np.where(np.logical_and(vbAnomal == False, vbDetects)):
            try:
                ax.plot([i, i], [-0.75, 1.5], "r--", lw=3, alpha=0.5)
            except ZeroDivisionError:
                pass

        plt.show()

    return {
        "vnAnomStarts": vnAnomStarts,
        "vnAnomEnds": vnAnomEnds,
        "viDetects": viDetects,
        "vbDetects": vbDetects,
        "vbAnomal": vbAnomal,
        "lbDetectedAnom": lbDetectedAnom,
        "lnFalseDetects": lnFalseDetects,
        "nTruePos": nTruePos,
        "nFalseNeg": nFalseNeg,
        "nTrueNeg": nTrueNeg,
        "nFalseDetects": nFalseDetects,
        "nFalseDetectIntervals": nFalseDetectIntervals,
        "fSensitivity": fSensitivity,
        "fSpecificity": fSpecificity,
        "nErrors": nErrors,
        "dSymptoms": dSymptoms,
    }


def find_detects(vSignal, fThrDetect, nClose):
    """
    find_detects - Find all threshold crossings from below in a 1D output
                   signal that count as detects. Return their indices
    """
    assert vSignal.ndim == 1, "Array dimension must be 1"

    vbAboveThr = vSignal > fThrDetect

    # Indices where threshold is crossed from below
    viDetects, = np.where(
        np.logical_and(
            vbAboveThr,  # Above threshold at current point
            np.r_[False, vbAboveThr[:-1]] == False,
        )
    )  # Below threshold at previous point

    # If Threshold crossed from below within nClose time steps after previous
    # crossing from below, do not count as additional detection
    viIgnoreDetects = viDetects - np.r_[-nClose, viDetects[:-1]] < nClose
    viDetects = viDetects[viIgnoreDetects == False]

    return viDetects


def find_detects_multi(mfSignal, vfThr, nClose):
    """
    find_detects_multi - Find all threshold crossings from below in a 2D output
                         signal that count as detects. Return their indices.
                         Then find the detects of all outputs taken together.
    """

    assert mfSignal.ndim == 2, "Array dimension must be 2"

    mbAboveThr = mfSignal > vfThr

    # - Boolean 2D array indicating for each channel where threshold is crossed from below
    mbDetects = np.zeros_like(mfSignal)
    mbPreviousBelowThr = np.zeros_like(mbAboveThr, bool)
    mbPreviousBelowThr[1:] = (mbAboveThr[:-1] == False)
    mbDetects = mbAboveThr & mbPreviousBelowThr

    # - 1D array with indices of points where threshold is crossed from below in any channel
    viDetects1D, = np.where(mbDetects.any(axis=1))

    # - If Threshold crossed from below within nClose time steps after previous
    #   crossing from below, do not count as additional detection
    viIgnoreDetects = viDetects1D - np.r_[-nClose, viDetects1D[:-1]] < nClose

    return viDetects1D[viIgnoreDetects == False]


def anom_starts_ends(vTarget):
    viAnomStarts, = np.where(
        np.logical_and(vTarget == 1, np.r_[0, vTarget[:-1]] == 0)  # Current value is 1
    )  # Previous value is 0
    viAnomEnds = (
        np.where(
            np.logical_and(
                vTarget == 1, np.r_[vTarget[1:], 0] == 0  # Previous value is 1
            )
        )[0]
        + 1
    )  # Current value is 0
    return viAnomStarts, viAnomEnds


def anom_starts_ends_multi(mTarget, bClasses=False):
    mbAnomal = mTarget > 0.5

    mbPreviousAnomal = np.zeros_like(mTarget, bool)
    mbPreviousAnomal[1:] = mbAnomal[:-1]

    # - Boolean indicating first time point of anomaly for each channel
    mbAnomStarts = mbAnomal & (mbPreviousAnomal == False)
    # - Boolean indicating last time point of anomaly for each channel
    mbAnomEnds = (mbAnomal == False) & mbPreviousAnomal

    # - Determine to which anomaly class the target belongs
    #   Works only if no two anomalies start at the same point (which should be the case)
    #   Overlapping anomalies are labelled correctly
    if bClasses:
        lnClasses = [np.where(vbStartsCurrent)[0][0]
            for vbStartsCurrent in mbAnomStarts if vbStartsCurrent.any()
        ]

    # - Taking all channels together, indices
    viAnomStartsAll = np.where(mbAnomStarts.any(axis=1))[0]
    viAnomEndsAll = (
        np.where(mbAnomEnds.any(axis=1))[0] + 1
    )  # Indices of first time points after anomalies

    # - Make sure arrays have same length, also if last time point is anomal
    if viAnomEndsAll.size < viAnomStartsAll.size:
        viAnomEndsAll = np.r_[viAnomEndsAll, mTarget.shape[0]]

    if bClasses:
        return viAnomStartsAll, viAnomEndsAll, lnClasses
    else:
        return viAnomStartsAll, viAnomEndsAll


def detected_anoms(vfOutput, vTarget, vbDetects, fThrDetect, nWindow=200, nClose=100):
    ## -- Which anomalies have been detected

    assert (
        len(vfOutput) == len(vTarget) == len(vbDetects)
    ), "Output, target and detecs dimentions must match"

    nDuration = len(vTarget)

    # - Indices of starts and ends of anomalies
    viAnomStarts, viAnomEnds = anom_starts_ends(vTarget)

    # - Detected anomalies (a detect during anomaly or within nWindow time steps after it)
    #   List of bools, indicating for each anomlay if detected or not
    lbDetectedAnom = [
        vbDetects[min(s, nDuration - 1) : min(e, nDuration - 1)].any()
        for s, e in zip(viAnomStarts, viAnomEnds + nWindow)
    ]

    # - Anomalies that occur close together (with less than nClose time steps between)
    #   Only require mfOutput to be above threshold, no new detect
    viAnomStartsClose, = np.where(
        viAnomStarts - np.r_[-nClose, viAnomStarts[:-1]] < nClose
    )
    vbAboveThr = vfOutput > fThrDetect
    for i in viAnomStartsClose:
        if vbAboveThr[viAnomStarts[i] : min(viAnomEnds[i] + nWindow, nDuration)].any():
            lbDetectedAnom[i] = True

    return lbDetectedAnom


def detected_anoms_multi(mfOutput, mTarget, vbDetects, vfThr, nWindow=200, nClose=100):
    ## -- Which anomalies have been detected

    assert (
        len(mfOutput) == len(mTarget) == len(vbDetects)
    ), "Output, target and detecs dimentions must match"

    nDuration = len(mTarget)

    # - Indices of starts and ends of anomalies
    viAnomStarts, viAnomEnds = anom_starts_ends_multi(mTarget)

    # - Detected anomalies (a detect during anomaly or within nWindow time steps after it)
    #   List of bools, indicating for each anomlay if detected or not
    lbDetectedAnom = [
        vbDetects[s:e].any() for s, e in zip(viAnomStarts, viAnomEnds + nWindow)
    ]

    # - Anomalies that occur close together (with less than nClose time steps between)
    #   Only require mfOutput to be above threshold, no new detect
    viAnomStartsClose, = np.where(
        viAnomStarts - np.r_[-nClose, viAnomEnds[:-1]] < nClose
    )
    mbAboveThr = (mfOutput > vfThr).any(axis=1)
    for i in viAnomStartsClose:
        if mbAboveThr[viAnomStarts[i] : min(viAnomEnds[i] + nWindow, nDuration)].any():
            lbDetectedAnom[i] = True

    return lbDetectedAnom


def anomal_points(mTarget, nWindow):

    mbAnomal = np.zeros_like(mTarget)

    if mTarget.ndim == 2:
        for iChannel in range(mTarget.shape[1]):
            # - Indices of starts and ends of anomalies
            viAnomStarts, viAnomEnds = anom_starts_ends(mTarget[:, iChannel])

            for iStart, iEnd in zip(viAnomStarts, viAnomEnds + nWindow):
                mbAnomal[iStart:iEnd, iChannel] = True

    else:
        raise ValueError("mTarget must be 2D array")

    return mbAnomal


def detects_in_normal_multi(mTarget, vbDetects, nWindow):
    """
    detects_in_normal_multi - Create list with numbers of detects for 
                              each normal interval. Only intervals 
                              where there is no anomaly in any output
                              dimension is considered normal
    :param mTarget:     2D np.ndarray Targets for all output dimensions
    :param vbDetects:   1D np.ndarray Boolean raster indicating time
                                      points with detects
    :param nWindow:     int Number of time points still assigned to
                            anomaly after its end
    :return:    lnFalseDetects  list of ints - Number of detects for each
                                normal interval
                viNormalStarts  1D np.ndarray Indices where normal intervals begin
                viNormalEnds    1D np.ndarray Indices where normal intervals end
    """

    mbAnomal = anomal_points(mTarget, nWindow)
    vbAnyTarget = mbAnomal.any(axis=1)

    return detects_in_normal(vbAnyTarget, vbDetects, nWindow)


def detects_in_normal(vTarget, vbDetects, nWindow):
    """
    detects_in_normal - Create list with numbers of detects for each
                        normal interval. Only consider one output dim.
    :param vTarget:     1D np.ndarray Targets for given output dimension
    :param vbDetects:   1D np.ndarray Boolean raster indicating time
                                      points with detects
    :param nWindow:     int Number of time points still assigned to
                            anomaly after its end
    :return:            list of ints Number of detects for each
                                     normal interval
    """

    # Starts and ends of normal intervals
    viNormalStarts = np.where(
        np.logical_and(vTarget == False, np.r_[True, vTarget[:-1]])
    )[0]
    viNormalEnds = np.where(
        np.logical_and(vTarget == False, np.r_[vTarget[1:], True])
    )[0] + 1
    

    # List with number of false detects for each interval
    nDuration = vTarget.shape[0]
    lnFalseDetects = [
        np.sum(vbDetects[s:e]) for s, e in zip(viNormalStarts, viNormalEnds)
    ]

    return lnFalseDetects, viNormalStarts, viNormalEnds


def errors_single(
    iChannel, mfOutput, mTarget, fThr, nWindow=200, nClose=100, bStrict=False, vfThr=None
):

    dErrors = dict()

    # - Detects, only count if the anomaly is detected by the specific channel
    #   It might happen, that another output would have detected the anomaly,
    #   but this is not counted here
    viDetects = find_detects(mfOutput[:, iChannel], fThr, nClose)
    vbDetects = np.zeros_like(mfOutput[:, iChannel], bool)
    vbDetects[viDetects] = True

    # - Which anomalies have been detected?
    dErrors['lbDetectedAnom'] = detected_anoms(
        mfOutput[:, iChannel],
        mTarget[:, iChannel],
        vbDetects,
        fThr,
        nWindow,
        nClose,
    )

    # - False negatives
    dErrors['nFalseNeg'] = (len(dErrors['lbDetectedAnom'])
        - np.sum(dErrors['lbDetectedAnom'])
    )

    if vfThr is not None:
        # - Detects are also counted if the anomaly has been detected by a different readout
        #   that has actually been trained for another anomaly type
        viDetectsAny = find_detects_multi(mfOutput, vfThr, nClose)
        vbDetectsAny = np.zeros_like(mfOutput[:, iChannel], bool)
        vbDetectsAny[viDetectsAny] = True

        dErrors['lbDetectedAnomAny'] = detected_anoms_multi(
            mfOutput,
            mTarget[:, iChannel].reshape(-1,1),
            vbDetects,
            vfThr,
            nWindow,
            nClose,
        )

        dErrors['nFalseNegAny'] = (len(dErrors['lbDetectedAnomAny'])
            - np.sum(dErrors['lbDetectedAnomAny'])
        )

    # - False positives

    if int(bStrict) == 0:
        # Do not count false detections if they just belong to different anomaly

        # Numbers of false detections during normal intervals
        dErrors['lnFalseDetects'], *__ = detects_in_normal_multi(
            mTarget, vbDetects, nWindow
        )
        # Number of normal intervals with one or more detections
        dErrors['nFalseDetectIntervals'] = np.sum(
            np.array(dErrors['lnFalseDetects']) > 0
        )

    elif int(bStrict) == 1:
        # Count false detections, also if they just belong to different anomaly

        # Numbers of false detections during normal intervals
        dErrors['lnFalseDetects'], *__ = detects_in_normal(
            mTarget[:, iChannel], vbDetects, nWindow
        )
        # Number of normal intervals with one or more detections
        dErrors['nFalseDetectIntervals'] = np.sum(
            np.array(dErrors['lnFalseDetects']) > 0
        )

    elif int(bStrict) == 2:
        
        # Numbers of false detections during normal intervals
        dErrors['lnFalseDetects'], *__ = detects_in_normal_multi(
            mTarget, vbDetects, nWindow
        )
        # Number of normal intervals with one or more detections
        dErrors['nFalseDetectIntervals'] = np.sum(
            np.array(dErrors['lnFalseDetects']) > 0
        )
       
        # Numbers of false detections during normal intervals
        dErrors['lnFalseDetectsStrict'], *__ = detects_in_normal(
            mTarget[:, iChannel], vbDetects, nWindow
        )
        # Number of normal intervals with one or more detections
        dErrors['nFalseDetectIntervalsStrict'] = np.sum(
            np.array(dErrors['lnFalseDetectsStrict']) > 0)

    return dErrors


def find_single_threshold_multi(
    iChannel: int,
    mOutput: np.ndarray,
    mTarget: np.ndarray,
    nWindow: int,
    nClose: int,
    nAttempts: int,
    fMin: float = 0,
    fMax: float = 2,
    nRecursions: int = 2,
    bStrict: bool = False,
    fErrorRatio: float = 1,
) -> (float, float):
    """Find a threshold that minimizes errors wrt vOutput and vTarget.
    Search on a grid with nAttempts steps. If nRecursions>0, continue
    search on grid around previous optimum.
        vOutput : Network output
        vTarget : Target output
        nWindow : Number of time steps after which an anomaly can still be detected
        nAttempts : Number of values to try for threshold within one run
        fMin, fMax : lowest, largest value of values to be tested
        nRecursions : How many times search should be refined on smaller-scale grid
        bStrict : bool, If False, false positives are only counted if 
                  there was no anomaly in any output target during the 
                  (false) detect. Otherwise only anomalies in  the 
                  target corresponding to iChannel are considered.
    """

    fFalseNeg = 2. * fErrorRatio / (fErrorRatio + 1)
    fFalseDet = 2. / (fErrorRatio + 1)

    def calc_errors(fThr):
        """
        calc_errors - Return in how many errors a given threshold results
        :param fThr:    float Threshold to be tested
        :return:        int Number of errors
        """
        dErrors = errors_single(
            iChannel, mOutput, mTarget, fThr, nWindow, nClose, bStrict
        )
        return (
            fFalseNeg * dErrors["nFalseNeg"]
            + fFalseDet * dErrors["nFalseDetectIntervals"]
        )

    vGrid = np.linspace(fMin, fMax, nAttempts)
    vErrors = np.array([calc_errors(fNewThr) for fNewThr in vGrid])

    # - Smallest error and corresponding threshold
    nIndMin = np.argmin(vErrors)
    fErrMin = vErrors[nIndMin]
    fThrOpt = vGrid[nIndMin]

    if nRecursions > 0:
        return find_single_threshold_multi(
            iChannel,
            mOutput,
            mTarget,
            nWindow,
            nClose,
            nAttempts,
            fMin=vGrid[max(0, nIndMin - 1)],
            fMax=vGrid[min(nIndMin + 2, nAttempts - 1)],
            nRecursions=nRecursions - 1,
            bStrict=bStrict,
            fErrorRatio=fErrorRatio,
        )
    else:
        # print('Best threshold found: {} ({} errors)'.format(fThrOpt, vErrors[nIndMin]))
        # - return best threshold and corresponding error
        return fThrOpt, fErrMin


def find_all_thresholds_multi(
    mOutput,
    mTarget,
    nWindow,
    nClose,
    nAttempts,
    fMin=0,
    fMax=2,
    nRecursions=2,
    bStrict=False,
    fErrorRatio: float = 1,
):
    def find_threshold_wrapper(iChannel):
        return find_single_threshold_multi(
            iChannel,
            mOutput,
            mTarget,
            nWindow,
            nClose,
            nAttempts,
            fMin,
            fMax,
            nRecursions,
            bStrict,
            fErrorRatio,
        )

    return np.array(
        [find_threshold_wrapper(iChannel) for iChannel in range(mOutput.shape[1])]
    )


def analyze_multi(
    mfOutput,
    mTarget,
    vInput,
    vfThr,
    nWindow=200,
    nClose=100,
    bVerbose=True,
    bPlot=True,
    bStrict=False,
):
    """Analyze output signal mfOutput wrt to target:
        - Sensitivity: Which of the anomalies have been detected?
            Detected means that there is at least one detection during
              or up to nWindow time steps after the anomaly
            A detection is when mfOutput crosses the threshold from below
              and there has been no detection within the last nClose
              time steps
            If two anomalies happen with less than
              nClose time steps between their beginning, detecting
              both is unlikely. Therefore in this case it is sufficient
              if mfOutput is above threshold.
        - False detections: Each detection that happens within a normal
          interval counts as false detection.
            A normal interval is an interval where no anomaly has
              occured within the previous nWindow time steps.
        - For the specificity, each normal interval in which one or
          more detections happen counts as false positive
    """

    # - Normal and anomal intervals
    mbAnomal = anomal_points(mTarget, nWindow)
    vbAllNormal = mbAnomal.any(axis=1) == False

    # - Starts and ends of anomalies (considers overlaps as separate anomalies)
    viAnomStarts, viAnomEnds, lnClassTarget = anom_starts_ends_multi(mTarget, bClasses=True)

    # - Register detects
    viDetects = find_detects_multi(mfOutput, vfThr, nClose)
    vbDetects = np.zeros(len(mfOutput), bool)
    vbDetects[viDetects] = True

    # - List with number of false detects for each interval
    lnFalseDetects, viNormalStarts, viNormalEnds = detects_in_normal_multi(
        mTarget, vbDetects, nWindow
    )

    # - Number of normal and abnormal intervals (all symptoms together)
    nNumNormal = viNormalStarts.size
    #   Anomalies in between normal intervals
    nNumAbnormal = viNormalStarts.size - 1
    #   Anomaly before first normal interval
    if viNormalStarts[0] != 0:
        nNumAbnormal += 1
    #   Anomaly after last normal interval
    if viNormalEnds[-1] != len(vbAllNormal):
        nNumAbnormal += 1

    # - Indices of false detects
    viFalseDetects = np.where(vbDetects & vbAllNormal)

    # - Specificity
    nTrueNeg = np.sum(np.asarray(lnFalseDetects) == 0)
    nFalseDetects = np.sum(lnFalseDetects)
    nFalseDetectIntervals = (
        len(lnFalseDetects) - nTrueNeg
    )  # Number of nomral intervals with one or more wrong detections
    fSpecificity = nTrueNeg / len(
        lnFalseDetects
    )  # len(lnFalseDetects) corresponds to number of normal intervals

    # - List of bools, indicating for each anomaly if it has been detected
    lbDetectedAnom = detected_anoms_multi(
        mfOutput, mTarget, vbDetects, vfThr, nWindow, nClose
    )

    # - Sensitivity
    
    nTruePos = np.sum(lbDetectedAnom)
    nFalseNeg = len(lbDetectedAnom) - nTruePos
    fSensitivity = nTruePos / len(lbDetectedAnom)

    nErrors = nFalseNeg + nFalseDetectIntervals


    # - Scale output dimensions according to their thresholds
    mfOutputScaled = np.zeros_like(mfOutput)
    mfOutputScaled[:, vfThr > 1e-6] = mfOutput[:, vfThr > 1e-6] / vfThr[vfThr > 1e-6]
    

    # - Classification

    # Classify detects according to maximum scaled output
    lnClass = [np.unravel_index(np.argmax(mfOutputScaled[iStart:iEnd]),
                                mfOutputScaled[iStart:iEnd].shape)[1]
        for iStart, iEnd in zip(viAnomStarts, viAnomEnds)
    ]

    # Indicate classes only for detected anomalies
    lnClassDetected = [nClass if bDetected else np.nan
        for nClass, bDetected in zip(lnClass, lbDetectedAnom)
    ]

    # Target classes
    lbClassified = [nDetected == nTarget for nDetected, nTarget in zip(lnClassDetected, lnClassTarget)]

    # Classification accuracy overall
    fClassified = np.sum(lbClassified) / len(lbClassified)

    # Classification accuracy, anomaly-wise
    lfClassifiedSpecific = [np.sum(np.array(lbClassified)[np.array(lnClassTarget) == iClass]
        ) / np.sum(np.array(lnClassTarget) == iClass) 
        if np.sum(np.array(lnClassTarget) == iClass) > 0 else np.nan
        for iClass in range(np.amax(lnClassTarget))
    ]

    # - Single symptom analysis
    def analyze_single(iChannel):
        dAnalysis = errors_single(
            iChannel, mfOutput, mTarget, vfThr[iChannel], nWindow, nClose, bStrict=2, vfThr=vfThr
        )

        # - Specificities
        dAnalysis["fSpecificity"] = (
            len(dAnalysis["lnFalseDetects"]) - dAnalysis["nFalseDetectIntervals"]
        ) / len(dAnalysis["lnFalseDetects"])

        dAnalysis["fSpecificityStrict"] = (
            len(dAnalysis["lnFalseDetectsStrict"]) - dAnalysis["nFalseDetectIntervalsStrict"]
        ) / len(dAnalysis["lnFalseDetectsStrict"])

        # - Sensitivity
        nNumAbnormalCurrent = len(dAnalysis["lbDetectedAnom"])
        if nNumAbnormalCurrent == 0:
            dAnalysis["fSensitivity"] = dAnalysis["fSensitivityAny"] = 1
        
        else:
            dAnalysis["fSensitivity"] = (
                nNumAbnormalCurrent - dAnalysis["nFalseNegAny"]
            ) / nNumAbnormalCurrent

            dAnalysis["fSensitivityStrict"] = (
                nNumAbnormalCurrent - dAnalysis["nFalseNeg"]
            ) / nNumAbnormalCurrent

        # - Error counts
        dAnalysis["nErrors"] = (
            dAnalysis["nFalseDetectIntervals"] + dAnalysis["nFalseNeg"]
        )
        
        dAnalysis["nErrors"] = (
            dAnalysis["nFalseDetectIntervalsStrict"] + dAnalysis["nFalseNeg"]
        )

        return dAnalysis

    dDetailled = {
        iChannel: analyze_single(iChannel) for iChannel in range(mfOutput.shape[1])
    }

    # - 1D-Signals
    vfOutMax = np.max(mfOutputScaled, axis=1)
    vfOutMean = np.mean(np.abs(mfOutputScaled), axis=1)

    if bVerbose:
        print(
            "Sensitivity : {:.1%} ({} of {} anomalies detected)".format(
                fSensitivity, nTruePos, len(lbDetectedAnom)
            )
        )
        print(
            "Specificity : {:.1%} ({} of {} normal intervals correct)".format(
                fSpecificity, nTrueNeg, len(lnFalseDetects)
            )
        )
        print(
            "{} Errors: \n {} of {} anomalies missed,".format(
                nErrors, nFalseNeg, len(lbDetectedAnom)
            )
            + " {} false detections (in {} intervals)".format(
                nFalseDetects, nFalseDetectIntervals
            )
        )

    if bPlot:

        # - Plotting

        fig, ax = plt.subplots()

        # Input
        ax.plot(0.2 * vInput, lw=2, color="k", alpha=0.4, zorder=-1)
        # Maximum scaled output
        ax.plot(np.clip(vfOutMax, -2, 2))
        # Target
        ax.plot(mbAnomal.any(axis=1), zorder=2)
        # Threshold
        ax.plot([0, len(vInput)], [1,1], 'k--', lw=2, zorder=10)

        # Higlight missed anomalies red, detected anomalies green and false positive intervals red
        for i, (s, e) in enumerate(zip(viAnomStarts, viAnomEnds)):
            strColorHighlight = "g" if lbDetectedAnom[i] else "r"
            ax.plot(
                [s, e + nWindow], [1.5, 1.5], color=strColorHighlight, lw=15, alpha=0.5
            )

        # Highlight normal intervals with wrong detection
        for i, (s, e) in enumerate(zip(viNormalStarts, viNormalEnds)):
            if lnFalseDetects[i] > 0.1:
                ax.plot([s, e], [-0.75, -0.75], color="r", lw=15, alpha=0.5)

        # Mark false detections with vertical red line
        for i in viFalseDetects:
            try:
                ax.plot([i, i], [-0.75, 1.5], "r--", lw=3, alpha=0.5)
            except ZeroDivisionError:
                pass

        plt.show()

    return {
        "viAnomStarts": viAnomStarts,  # Starts of anomalies, considering overlaps as separate anomalies
        "viAnomEnds": viAnomEnds,  # Ends of anomalies, considering overlaps as separate anomalies
        "viNormalStarts": viNormalStarts,  # Starts of normal intervals
        "viNormalEnds": viNormalEnds,  # Ends of normal intervals
        "vbDetects": vbDetects,  # Boolean array indicating at which time points there are detects
        "lbDetectedAnom": lbDetectedAnom,  # List of bools indicating which anomalies have been detected
        "lnFalseDetects": lnFalseDetects,  # List of ints indicating number of false detections during each normal interval
        "nTruePos": nTruePos,  # True positives
        "nFalseNeg": nFalseNeg,  # False Negatives
        "nTrueNeg": nTrueNeg,  # True Negatives
        "nFalseDetects": nFalseDetects,  # Number of false detections
        "nFalseDetectIntervals": nFalseDetectIntervals,  # Number of normal intervals with at least one false detection
        "fSensitivity": fSensitivity,
        "fSpecificity": fSpecificity,
        "nErrors": nErrors,
        "dSymptoms": dDetailled,  # Details on single symptom types
        "vfOutMean": vfOutMean,
        "vfOutMax": vfOutMax,
        "lnClass" : lnClass,
        "lnClassDetected" : lnClassDetected,
        "lnClassTarget" : lnClassTarget,
        "lbClassified" : lbClassified,
        "fClassified" : fClassified,
        "lfClassifiedSpecific" : lfClassifiedSpecific,
    }


### --- Analyse reservoir activities


def mean_rates(mfSpikeRaster, tDt=1):
    return np.mean(mfSpikeRaster, axis=0) / tDt


def filter_trains_exp(mfSpikeRaster, tTau, tDt=1):

    # - Normalized exponential kernel
    vfKernel = np.exp(-np.arange(0, int(10 * tTau), tDt) / tTau) * (tDt / tTau)
    #   Add leading 0 for causality
    vfKernel = np.r_[0, vfKernel]

    return filter_spike_raster(vfKernel, mfSpikeRaster)


def filter_trains_box(mfSpikeRaster, tWindow, tDt=1):

    # - Box-shaped filter
    vfBox = np.ones(int(tWindow / tDt)) * tDt / tWindow
    #   Leading 0 for causality
    vfBox = np.r_[0, vfBox]

    return filter_spike_raster(vfBox, mfSpikeRaster)

def filter_exp_box(mfSpikeRaster, tTau, tWindow, tDt=1):

    ### --- Exptl filtering
    # - Normalized exponential kernel
    vfKernel = np.exp(-np.arange(0, int(10 * tTau), tDt) / tTau) * (tDt / tTau)
    #   Add leading 0 for causality
    vfKernel = np.r_[0, vfKernel]

    mfFilteredExp = filter_spike_raster(vfKernel, mfSpikeRaster)

    ### --- Moving window
    # - Box-shaped filter
    vfBox = np.ones(int(tWindow / tDt)) * tDt / tWindow

    return filter_spike_raster(vfBox, mfFilteredExp)    


def filter_spike_raster(vfKernel, mfSpikeRaster):
    # - Filter spike trains
    mfFiltered = np.zeros(
        (len(vfKernel) - 1 + mfSpikeRaster.shape[0], mfSpikeRaster.shape[1])
    )
    for iChannel, vEvents in enumerate(mfSpikeRaster.T):
        mfFiltered[:, iChannel] = fftconvolve(vfKernel, vEvents)

    return mfFiltered

def interspike_intervals(tsReservoir, tDt, vnSelectChannels=None):
    
    nDim = np.amax(tsReservoir.vnChannels) + 1

    # - Get rasterized spike trains
    vtTimeBase, __, mfSpikeRaster, __ = tsReservoir.raster(
        tDt=tDt, vnSelectChannels=vnSelectChannels, bSamples=False
    )

    mfISI = np.zeros_like(mfSpikeRaster, float)

    for iChannel in range(nDim):
        # - Spike locations in raster
        viSpikeInds = np.where(mfSpikeRaster[:,iChannel])[0]

        # - Interspike intervals (in numbers of time steps)
        vnISI = np.diff(viSpikeInds)
        
        # - Start first interval after first spike, last interval ends with last spike
        nStart = viSpikeInds[0] + 1
        nEnd = viSpikeInds[-1] + 1
        # - Fill in inter spike intervals
        mfISI[nStart : nEnd, iChannel] = np.repeat(vnISI * tDt, vnISI)
        # - Fill entries before first and after last interval with nan
        mfISI[ : nStart, iChannel] = np.nan
        mfISI[ nEnd : , iChannel] = np.nan

    return mfISI



def plot_activities_2d(tsReservoir, tTau, tDt=None, strKernel='exp', vnSelectChannels=None):
    if tsReservoir.vnChannels.size == 0:
        print("No Spikes recorded")
        return

    if tDt is None:
        tDt = tTau / 10

    nDim = np.amax(tsReservoir.vnChannels) + 1

    # - Get rasterized spike trains
    __, __, mfSpikeRaster, __ = tsReservoir.raster(
        tDt=tDt, vnSelectChannels=vnSelectChannels, bSamples=False
    )

    # - Filter spike trains
    if strKernel == 'exp':
        mfFiltered = filter_trains_exp(mfSpikeRaster, tTau, tDt)
    elif strKernel == 'box':
        mfFiltered = filter_trains_box(mfSpikeRaster, tTau, tDt)
    else:
        print('Kernel type not knwon')
        return

    vtTime = np.arange(len(mfFiltered)) * tDt + tsReservoir.tStart
    
    fig = plt.figure()
    plt.plot(vtTime, mfFiltered)

    return fig


def plot_activities(tsReservoir, tTau, tDt=None, vnSelectChannels=None):

    if tsReservoir.vnChannels.size == 0:
        print("No Spikes recorded")
        return

    if tDt is None:
        tDt = tTau / 10

    nDim = np.amax(tsReservoir.vnChannels) + 1

    # - Get rasterized spike trains
    vtTimeBase, __, mfSpikeRaster, __ = tsReservoir.raster(
        tDt=tDt, vnSelectChannels=vnSelectChannels, bSamples=False
    )

    # - Filter spike trains with exponentially decaying kernel

    # Define exponential kernel (Cut after 10*tTau, where it is only around 5e-5)
    vfKernel = np.exp(-np.arange(0, 10 * tTau, tDt) / tTau)
    # - Make sure spikes only have effect on next time step
    vfKernel = np.r_[0, vfKernel]

    # - Apply kernel to spike trains
    mfFiltered = np.zeros((len(vtTimeBase) + len(vfKernel) - 1, nDim))
    for channel, vEvents in enumerate(mfSpikeRaster.T):
        vfConv = fftconvolve(vEvents, vfKernel, "full")
        mfFiltered[:, channel] = vfConv

    # - Prepare axis
    viNeurons = np.arange(nDim)
    vtTime = vtTimeBase[0] + np.arange(mfFiltered.shape[0]) * tDt
    X, Y = np.meshgrid(viNeurons, vtTime)

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    ax.plot_surface(X, Y, mfFiltered)

    return ax


def find_threshold(
    vOutput,
    vTarget,
    nWindow,
    nClose,
    nAttempts,
    fMin=0,
    fMax=1,
    nRecursions=2,
    fErrorRatio=1,
):
    """Find a threshold that minimizes errors wrt vOutput and vTarget.
    Search on a grid with nAttempts steps. If nRecursions>0, continue
    search on grid around previous optimum.
        vOutput : Network output
        vTarget : Target output
        nWindow : Number of time steps after which an anomaly can still be detected
        nAttempts : Number of values to try for threshold within one run
        fMin, fMax : lowest, largest value of values to be tested
        nRecursion : How many times search should be refined on smaller-scale grid
    """

    vGrid = np.linspace(fMin, fMax, nAttempts)

    fFalseNeg = 2. * fErrorRatio / (fErrorRatio + 1)
    fFalseDet = 2. / (fErrorRatio + 1)

    def calc_error(fThr):
        dAnalysis = analyze(vOutput, vTarget, None, fThr, nWindow, nClose, False, False)
        return (
            fFalseDet * dAnalysis["nFalseDetectIntervals"]
            + fFalseNeg * dAnalysis["nFalseNeg"]
        )

    vErrors = np.array([calc_error(fThr) for fThr in vGrid])
    nIndMin = np.argmin(vErrors)

    fThrOpt = vGrid[nIndMin]
    # print('Best threshold so far: {} ({} errors) \nRemaining recursions: {}'.format(
    #     fThrOpt, vErrors[nIndMin], nRecursions))
    if nRecursions > 0:
        return find_threshold(
            vOutput,
            vTarget,
            nWindow,
            nClose,
            nAttempts,
            fMin=vGrid[max(0, nIndMin - 1)],
            fMax=vGrid[min(nIndMin + 2, nAttempts - 1)],
            nRecursions=nRecursions - 1,
            fErrorRatio=fErrorRatio,
        )
    else:
        # print('Best threshold found: {} ({} errors)'.format(fThrOpt, vErrors[nIndMin]))
        return fThrOpt


def find_coincidental_normals(vInput0, mTarget0, mTarget1, tMeanDur):
    """Check if by coincidence there are unmarked, normal ecg rhythms
    in the input. This can be useful if the input signal is constructed
    from many single segments (instead of complete rhythms). Complete
    rhythms will be detected as 'possibly normal' unless they are
    bradycarid or tachycardic or if anomalies are labelld in detail in
    second stage (bDetailledAnomalies1 == True).
        vInput0 : Input to first reservoir
        mTarget0 : Labels for first reservoir in binary matrix form
        mTarget1 : Labels for second reservoir in binary matrix form
        tMeanDur : Expected duration of a normal rhythm (in time steps)
    """

    vTarget0Int = np.where(mTarget0)[1]  # mTarget0 in (non-binary) vector form
    vTarget1Int = np.where(mTarget1)[1]  # mTarget0 in (non-binary) vector form
    vSegments, vSTInds, vSTReps, vSTRec = remove_consecutive_duplicates(vTarget0Int)
    vNormalSequence = np.array([0, 1, 0, 2, 0, 3, 0])
    nLenNS = len(vNormalSequence)
    mIndices = np.arange(len(vSegments) - nLenNS + 1)[:, None] + np.arange(nLenNS)
    # vFound is True if at the corresponding position in vSegments an occurence of vNormalSequence begins
    vFound = (vSegments[mIndices] == vNormalSequence).all(1)
    vFound = np.r_[vFound, (nLenNS - 1) * [False]]
    # Indices where vInput0 corresponds to vNormalSequence
    lvIndFound = [
        np.arange(
            vSTInds[min(i, len(vSTInds) - 1)],
            vSTInds[min(i + nLenNS, len(vSTInds) - 1)],
        )
        for i in np.where(vFound)[0]
    ]
    # Corresponding values
    lvMatches = [vInput0[i] for i in lvIndFound]
    # Iterate over these matches. If Their length corresponds to a normal ecg rhythm and they are
    # not tagged as such, plot them and ask if they should be tagged as normal
    lIndsUnflagged = []
    for i, (vMatch, vInd) in enumerate(zip(lvMatches, lvIndFound)):
        if (
            0.75 * tMeanDur <= len(vMatch) <= 1.4 * tMeanDur
            and not vTarget1Int[vInd].any()
        ):
            # plt.plot(vMatch)
            # plt.show()
            # b.append(input('If this looks like a correct ecg rhythm press 1 and then enter, otherwise just enter \n'))
            # plt.close()
            lIndsUnflagged.append(i)
    print(
        "There are {}  unflagged but possibly normal rhythms ({:.1%})".format(
            len(lIndsUnflagged), len(lIndsUnflagged) / len(lvMatches)
        )
    )

    return np.array(lvMatches)[lIndsUnflagged], np.array(lvIndFound)[lIndsUnflagged]


def remove_consecutive_duplicates(vA):
    """All except one element of each sequence of consecutive
    duplicates in an array are removed. Apart from that the original
    structure is maintained. For instance [a,a,a,b,b,c,a,a,d] becomes
    [a,b,c,a,d].
    Return the resulting sequence vValues, the (first) indices vIndices
    of these elements in the array, the number of repetitions for each
    vRepetitions element and the indices of the original array elements
    within vValues, so that vValues = vA[vIndices]. vA can be
    reconstructed with vA = vValues[vReconstruct] or
    vA = np.array([v for v, r in zip(vValues, vRepetitions) for i in range(r)])
    """

    # - Only keep elements that are non equal to the following one
    vB = np.r_[vA[1:], np.nan]
    vC = np.r_[np.nan, vA[:-1]]
    vIndices = np.where(vA != vC)[0]
    vValues = vA[vIndices]
    vRepetitions = np.diff(np.r_[vIndices, len(vA)])
    vReconstruct = np.array([j for j, r in enumerate(vRepetitions) for i in range(r)])
    return vValues, vIndices, vRepetitions, vReconstruct


def plot(vInput, dOut, ddAn, dfThr, fDt, sTitle=""):

    dAnLoc = ddAn["local"]
    dAnBrd = ddAn["broad"]
    dAnFix = ddAn["fix"]

    fig = bk.figure(
        title=sTitle,
        plot_height=400,
        x_axis_label="Time in s",
        y_axis_label="Output",
        x_range=[0, 5],
        y_range=[-1.5, 2],
        tools="pan, xwheel_zoom, xwheel_pan, box_zoom, reset, save",
    )
    fig.line(
        fDt * np.arange(len(vInput)),
        0.3 * vInput,
        legend="Input",
        color="grey",
        alpha=0.5,
        line_width=3,
    )
    fig.line(
        fDt * np.arange(len(dOut["local"])),
        dOut["local"].flatten(),
        legend="Only anomalies labelled",
        color="blue",
    )
    fig.line(
        fDt * np.arange(len(dOut["broad"])),
        dOut["broad"].flatten(),
        legend="Complete rhythms labelled",
        color="orange",
    )
    fig.line(
        fDt * np.arange(len(dOut["fix"])),
        dOut["fix"].flatten(),
        legend="Fix length after onset",
        color="maroon",
    )
    # Indicate anomalies
    mAnomalyStartEndLoc = (
        fDt * np.vstack((dAnLoc["vnAnomStarts"], dAnLoc["vnAnomEnds"])).T
    )
    mAnomalyStartEndBrd = (
        fDt * np.vstack((dAnBrd["vnAnomStarts"], dAnBrd["vnAnomEnds"])).T
    )
    mAnomalyStartEndFix = (
        fDt * np.vstack((dAnFix["vnAnomStarts"], dAnFix["vnAnomEnds"])).T
    )
    # Hit by compl rhythm
    fig.multi_line(
        mAnomalyStartEndBrd[dAnBrd["lbDetectedAnom"]].tolist(),
        (-0.5 * np.ones((np.sum(dAnBrd["lbDetectedAnom"]), 2))).tolist(),
        line_width=38,
        alpha=0.5,
        color="green",
    )
    # Missed by compl rhythm
    fig.multi_line(
        mAnomalyStartEndBrd[np.array(dAnBrd["lbDetectedAnom"]) == False].tolist(),
        (
            -0.5 * np.ones((np.sum(np.array(dAnBrd["lbDetectedAnom"]) == False), 2))
        ).tolist(),
        line_width=38,
        alpha=0.5,
        color="red",
    )
    # Hit by fix interval
    fig.multi_line(
        mAnomalyStartEndFix[dAnFix["lbDetectedAnom"]].tolist(),
        (-1 * np.ones((np.sum(dAnFix["lbDetectedAnom"]), 2))).tolist(),
        line_width=38,
        alpha=0.5,
        line_dash=(12, 4),
        color="green",
    )
    # Missed by fix interval
    fig.multi_line(
        mAnomalyStartEndFix[np.array(dAnFix["lbDetectedAnom"]) == False].tolist(),
        (
            -1 * np.ones((np.sum(np.array(dAnFix["lbDetectedAnom"]) == False), 2))
        ).tolist(),
        line_width=38,
        alpha=0.5,
        line_dash=(12, 4),
        color="red",
    )
    # Hit by local label
    fig.multi_line(
        mAnomalyStartEndLoc[dAnLoc["lbDetectedAnom"]].tolist(),
        (1.5 * np.ones((np.sum(dAnLoc["lbDetectedAnom"]), 2))).tolist(),
        line_width=38,
        alpha=0.5,
        # line_dash=(12,4),
        color="green",
    )
    # Missed by local label
    fig.multi_line(
        mAnomalyStartEndLoc[np.array(dAnLoc["lbDetectedAnom"]) == False].tolist(),
        (
            1.5 * np.ones((np.sum(np.array(dAnLoc["lbDetectedAnom"]) == False), 2))
        ).tolist(),
        line_width=38,
        alpha=0.5,
        # line_dash=(12,4),
        color="red",
    )
    # False detections local label
    vnFalseDetectsLoc = (
        fDt
        * np.where(np.logical_and(dAnLoc["vbDetects"], dAnLoc["vbAnomal"] == False))[0]
    )
    fig.multi_line(
        np.vstack((vnFalseDetectsLoc, vnFalseDetectsLoc)).T.tolist(),
        np.vstack(
            (2 * np.ones_like(vnFalseDetectsLoc), 0.5 * np.ones_like(vnFalseDetectsLoc))
        ).T.tolist(),
        color="red",
        line_dash="dashed",
        line_width=3,
    )
    # False detections fix interval
    vnFalseDetectsFix = (
        fDt
        * np.where(np.logical_and(dAnFix["vbDetects"], dAnFix["vbAnomal"] == False))[0]
    )
    fig.multi_line(
        np.vstack((vnFalseDetectsFix, vnFalseDetectsFix)).T.tolist(),
        np.vstack(
            (
                -0.5 * np.ones_like(vnFalseDetectsFix),
                -1.5 * np.ones_like(vnFalseDetectsFix),
            )
        ).T.tolist(),
        color="red",
        line_dash="dashed",
        line_width=3,
    )
    # False detections  compl rhythm
    vnFalseDetectsBrd = (
        fDt
        * np.where(np.logical_and(dAnBrd["vbDetects"], dAnBrd["vbAnomal"] == False))[0]
    )
    fig.multi_line(
        np.vstack((vnFalseDetectsBrd, vnFalseDetectsBrd)).T.tolist(),
        np.vstack(
            (
                0.5 * np.ones_like(vnFalseDetectsBrd),
                -0.5 * np.ones_like(vnFalseDetectsBrd),
            )
        ).T.tolist(),
        color="red",
        line_dash="dashed",
        line_width=3,
    )
    # Thresholds
    fig.line(
        [0, fDt * len(vInput)],
        [dfThr["local"], dfThr["local"]],
        line_dash="dashed",
        color="blue",
        legend="Thr Local",
        line_width=2,
    )
    fig.line(
        [0, fDt * len(vInput)],
        [dfThr["broad"], dfThr["broad"]],
        line_dash="dashed",
        color="orange",
        legend="Thr Compl.",
        line_width=2,
    )
    fig.line(
        [0, fDt * len(vInput)],
        [dfThr["fix"], dfThr["fix"]],
        line_dash="dashed",
        color="maroon",
        legend="Thr Fix Int.",
        line_width=2,
    )

    return fig


class PlotResultsAnomalies:
    def __init__(self, dInputs, dResultsAn, ddAnalysisAnom, dThr, fDt):
        dsAnomaliesSL = {
            "complete_Tinv": "Inverted T wave",
            "complete_Pinv": "Inverted P wave",
            "complete_noP": "Missing P wave",
            "complete_STelev": "Elevated ST segment",
            "complete_STdepr": "Depressed ST segment",
            "complete_noQRS": "Missing QRS complex",
            "complete_tach": "Tachycardia",
            "complete_brad": "Bradycardia",
        }
        lsAnomalies = list(dInputs.keys())
        lsNets = list(dThr.keys())
        ddOutputs = {
            ano: {net: dResultsAn[(net, ano)] for net in lsNets} for ano in lsAnomalies
        }
        ddAnalysis = {
            ano: {net: ddAnalysisAnom[(net, ano)] for net in lsNets}
            for ano in lsAnomalies
        }
        self.ddAnalysisAnom = ddAnalysisAnom
        self.dBkPlots = {
            ano: plot(
                dInputs[ano],
                ddOutputs[ano],
                ddAnalysis[ano],
                dThr,
                fDt,
                sTitle=dsAnomaliesSL[ano],
            )
            for ano in lsAnomalies
        }

    def __call__(self, sAnom):
        """Print results for anomalies and return plot"""
        dAnLoc = self.ddAnalysisAnom[("local", sAnom)]
        dAnBrd = self.ddAnalysisAnom[("broad", sAnom)]
        dAnFix = self.ddAnalysisAnom[("fix", sAnom)]
        print("Only anomalies Labelled:")
        print(
            "Sensitivity : {:.1%} (in total: {} anomalies)".format(
                dAnLoc["fSensitivity"], len(dAnLoc["lbDetectedAnom"])
            )
        )
        print(
            "Specificity : {:.1%} (in total: {} normal intervals)".format(
                dAnLoc["fSpecificity"], len(dAnLoc["lnFalseDetects"])
            )
        )
        print(
            "{} Errors: \n {} of {} anomalies missed, {} false detections".format(
                dAnLoc["nErrors"],
                dAnLoc["nFalseNeg"],
                len(dAnLoc["lnFalseDetects"]),
                dAnLoc["nFalseDetects"],
            )
        )

        print("\nComplete rhythms labelled:")
        print(
            "Sensitivity : {:.1%} (in total: {} anomalies)".format(
                dAnBrd["fSensitivity"], len(dAnBrd["lbDetectedAnom"])
            )
        )
        print(
            "Specificity : {:.1%} (in total: {} normal intervals)".format(
                dAnBrd["fSpecificity"], len(dAnBrd["lnFalseDetects"])
            )
        )
        print(
            "{} Errors: \n {} of {} anomalies missed, {} false detections".format(
                dAnBrd["nErrors"],
                dAnBrd["nFalseNeg"],
                len(dAnBrd["lnFalseDetects"]),
                dAnBrd["nFalseDetects"],
            )
        )

        print("\nFixed length labels:")
        print(
            "Sensitivity : {:.1%} (in total: {} anomalies)".format(
                dAnFix["fSensitivity"], len(dAnFix["lbDetectedAnom"])
            )
        )
        print(
            "Specificity : {:.1%} (in total: {} normal intervals)".format(
                dAnFix["fSpecificity"], len(dAnFix["lnFalseDetects"])
            )
        )
        print(
            "{} Errors: \n {} of {} anomalies missed, {} false detections".format(
                dAnFix["nErrors"],
                dAnFix["nFalseNeg"],
                len(dAnFix["lnFalseDetects"]),
                dAnFix["nFalseDetects"],
            )
        )

        return self.dBkPlots[sAnom]
