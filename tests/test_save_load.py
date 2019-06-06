
def test_save_load():
    """ Test RecIAFNest"""
    from NetworksPython.layers import RecIAFSpkInNest, FFIAFNest, FFExpSynTorch
    from NetworksPython import timeseries as ts
    from NetworksPython.networks import network as nw
    from NetworksPython.networks import Network as nws
    import numpy as np
    import pylab as plt

    # - Generic parameters
    weights = np.ones([1, 1]) * 0.01
    weights_in = [[0.1, 0, 0]]

    weights_rec = [[0, 0.1, 0], [0, 0, 0.1], [0.0, 0, 0]]
    mfWOut = [[1], [1], [1]]
    vfBias = 0.0
    dt = 0.001
    vtTauN = 0.02
    vtTauS = 0.05
    vfVThresh = -0.055
    vfVReset = -0.065
    vfVRest = -0.065
    vfCapacity = 100.
    tRef = 0.001

    np.random.seed(0)

    fl0 = FFIAFNest(weights=weights,
                    dt=dt,
                    vfBias=vfBias,
                    vtTauN=vtTauN,
                    vfVReset=vfVReset,
                    vfVRest=vfVRest,
                    vfVThresh=vfVThresh,
                    vfCapacity=vfCapacity,
                    tRefractoryTime=tRef,
                    num_cores=1,
                    bRecord=True,
                    name="FF")

    fl1 = RecIAFSpkInNest(weights_in=weights_in,
                          weights_rec=weights_rec,
                          dt=dt,
                          vfBias=vfBias,
                          vtTauN=vtTauN,
                          vtTauS=vtTauS,
                          vfVThresh=vfVThresh,
                          vfVReset=vfVReset,
                          vfVRest=vfVRest,
                          vfCapacity=vfCapacity,
                          tRefractoryTime=tRef,
                          num_cores=1,
                          bRecord=True,
                          name="Rec")


    net0 = nw.Network(fl0, fl1)

    net0.save("test_nw.json")
    net1 = nws.load("test_nw.json")
    fl2 = net1.FF
    fl3 = net1.Rec

    np.random.seed(0)


    np.random.seed(0)

    # - Input signal
    vTime = np.arange(0, 1, dt)
    vVal = np.zeros([len(vTime), 1])
    vVal[200:201] = 0.25

    tsInCont = ts.TSContinuous(vTime, vVal)

    epsilon = 0.0000001


    # - Compare states before and after
    np.random.seed(0)
    dAct0 = net0.evolve(tsInCont, duration=1.0)

    np.random.seed(0)
    dAct1 = net1.evolve(tsInCont, duration=1.0)

    assert (np.abs(fl0.mfRecordStates - fl2.mfRecordStates) < epsilon).all()
    assert (np.abs(fl1.mfRecordStates - fl3.mfRecordStates) < epsilon).all()

