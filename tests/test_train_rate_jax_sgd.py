"""
Test training of rate-based Euler models in rate-jax.py
"""

import numpy as np
import pytest


def test_imports():
    from rockpool.layers.training import JaxTrainer


def test_train_rate_jax_sgd_RecRateEulerJax_IO():
    from rockpool.layers import RecRateEulerJax_IO
    from rockpool import TSContinuous

    # - Generic parameters
    w_in = 10.0 * np.ones((1, 2))
    w_recurrent = 10.0 * np.ones((2, 2))
    w_out = 10.0 * np.ones((2, 1))

    # - Layer generation
    fl0 = RecRateEulerJax_IO(
        w_in=w_in,
        w_recurrent=w_recurrent,
        w_out=w_out,
        bias=0.0,
        noise_std=0.0,
        tau=20.0,
        dt=1,
    )

    # - Define simple input and target
    ts_input = TSContinuous([0, 1, 2, 3], [1, 1, 1, 1], t_stop=4)
    ts_target = TSContinuous([0, 1, 2, 3], [0.1, 0.1, 0.1, 0.1])

    # - Initialise training
    l, g, output_fcn = fl0.train_output_target(
        ts_input, ts_target, is_first=True, debug_nans=True,
    )

    # - Test output function
    o = output_fcn()

    # - Test known loss and gradient values
    l_target = 6488
    g_target = {
        "bias": np.array([639, 639]),
        "tau": np.array([-637, -637]),
        "w_in": np.array([[639, 639]]),
        "w_out": np.array([[639], [639]]),
        "w_recurrent": np.array([[339, 339], [339, 339]]),
    }

    # - Check loss
    assert (
        np.abs(np.round(l) - l_target) < 1
    ), "MSE loss value does not match known target"

    # - Check gradients
    for k, v in g.items():
        assert np.all(np.abs(np.round(v) - g_target[k]) < 1), (
            "Gradient value for [" + k + "] does not match known target"
        )

    # - Perform intermediate training step
    fl0.train_output_target(ts_input, ts_target)

    # - Perform final training step
    fl0.train_output_target(ts_input, ts_target, is_last=True)

    target_eps = 1e-3

    state_target = [0.0, 0.0]
    tau_target = [20.000298, 20.000298]
    bias_target = [-0.0003, -0.0003]
    w_in_target = [[9.9997, 9.9997]]
    w_out_target = [[9.9997], [9.9997]]
    w_recurrent_target = [[9.9997, 9.9997], [9.9997, 9.9997]]

    assert np.all(np.abs(fl0.state - state_target) < target_eps)
    assert np.all(np.abs(fl0.tau - tau_target) < target_eps)
    assert np.all(np.abs(fl0.bias - bias_target) < target_eps)
    assert np.all(np.abs(fl0.w_in - w_in_target) < target_eps)
    assert np.all(np.abs(fl0.w_out - w_out_target) < target_eps)
    assert np.all(np.abs(fl0.w_recurrent - w_recurrent_target) < target_eps)

    # - Test known output values
    ts_output = fl0.evolve(ts_input, duration=3)

    ts_output_target = [[9.99895], [29.49645996], [67.51574707]]

    assert np.all(np.abs(ts_output.samples - ts_output_target) < target_eps)

    # - test batched training
    inp_batch = np.repeat(np.expand_dims(ts_input.samples, 0), 2, axis=0)
    tgt_batch = np.repeat(np.expand_dims(ts_target.samples, 0), 2, axis=0)
    fl0.train_output_target(
        inp_batch, tgt_batch, is_first=True, debug_nans=True, batch_axis=0,
    )


def test_train_rate_jax_sgd_RecRateEulerJax_IO_aux_loss():
    from rockpool.layers import RecRateEulerJax_IO
    from rockpool import TSContinuous
    from rockpool.layers.training.gpl.jax_trainer import loss_mse_reg
    from time import time

    # - Generic parameters
    w_in = 10.0 * np.ones((1, 2))
    w_recurrent = 10.0 * np.ones((2, 2))
    w_out = 10.0 * np.ones((2, 1))

    # - Layer generation
    fl0 = RecRateEulerJax_IO(
        w_in=w_in,
        w_recurrent=w_recurrent,
        w_out=w_out,
        bias=0.0,
        noise_std=0.0,
        tau=20.0,
        dt=1,
    )

    # - Define simple input and target
    ts_input = TSContinuous([0, 1, 2, 3], [1, 1, 1, 1], t_stop=4)
    ts_target = TSContinuous([0, 1, 2, 3], [0.1, 0.1, 0.1, 0.1])

    # - Add some generic aux output to loss function
    class Loss:
        # - Each time this class is called it returns the MSE loss and an integer that grows iteratively
        def __init__(self):
            self.i = 0

        def __call__(self, *args, **kwargs):
            return self.loss_function(*args, **kwargs)

        def loss_function(
            self,
            params,
            states_t,
            output_batch_t,
            target_batch_t,
            min_tau,
            lambda_mse=1,
            reg_tau=10000,
            reg_l2_rec=1,
        ):
            self.i += 1
            return (
                loss_mse_reg(
                    params,
                    states_t,
                    output_batch_t,
                    target_batch_t,
                    min_tau,
                    lambda_mse,
                    reg_tau,
                    reg_l2_rec,
                ),
                self.i,
            )

    loss_inc = Loss()

    # - Initialise training
    l, g, output_fcn = fl0.train_output_target(
        ts_input,
        ts_target,
        is_first=True,
        debug_nans=True,
        loss_aux=True,
        loss_fcn=loss_inc,
        loss_params=fl0._default_loss_params,
    )

    # - Test output function
    o = output_fcn()

    # - Test known loss and gradient values
    l_target = 6488
    g_target = {
        "bias": np.array([639, 639]),
        "tau": np.array([-637, -637]),
        "w_in": np.array([[639, 639]]),
        "w_out": np.array([[639], [639]]),
        "w_recurrent": np.array([[339, 339], [339, 339]]),
    }

    # - Check loss
    assert (
        np.abs(np.round(l[0]) - l_target) < 1
    ), "MSE loss value does not match known target"
    print("Loss output:", l)

    # - Check gradients
    for k, v in g.items():
        assert np.all(np.abs(np.round(v) - g_target[k]) < 1), (
            "Gradient value for [" + k + "] does not match known target"
        )

    # - Perform intermediate training step
    fl0.train_output_target(ts_input, ts_target, loss_aux=True)

    # - Perform final training step
    fl0.train_output_target(ts_input, ts_target, is_last=True, loss_aux=True)

    target_eps = 1e-3

    state_target = [0.0, 0.0]
    tau_target = [20.000298, 20.000298]
    bias_target = [-0.0003, -0.0003]
    w_in_target = [[9.9997, 9.9997]]
    w_out_target = [[9.9997], [9.9997]]
    w_recurrent_target = [[9.9997, 9.9997], [9.9997, 9.9997]]

    assert np.all(np.abs(fl0.state - state_target) < target_eps)
    assert np.all(np.abs(fl0.tau - tau_target) < target_eps)
    assert np.all(np.abs(fl0.bias - bias_target) < target_eps)
    assert np.all(np.abs(fl0.w_in - w_in_target) < target_eps)
    assert np.all(np.abs(fl0.w_out - w_out_target) < target_eps)
    assert np.all(np.abs(fl0.w_recurrent - w_recurrent_target) < target_eps)

    # - Test known output values
    ts_output = fl0.evolve(ts_input, duration=3)

    ts_output_target = [[9.99895], [29.49645996], [67.51574707]]

    assert np.all(np.abs(ts_output.samples - ts_output_target) < target_eps)

    # - test batched training
    inp_batch = np.repeat(np.expand_dims(ts_input.samples, 0), 2, axis=0)
    tgt_batch = np.repeat(np.expand_dims(ts_target.samples, 0), 2, axis=0)
    fl0.train_output_target(
        inp_batch,
        tgt_batch,
        is_first=True,
        debug_nans=True,
        batch_axis=0,
        loss_fcn=Loss(),
        loss_params=fl0._default_loss_params,
        loss_aux=True,
    )


def test_train_rate_jax_sgd_FFRateEulerJax():
    from rockpool.layers import FFRateEulerJax
    from rockpool import TSContinuous

    # - Generic parameters
    w_in = 2 * np.random.rand(1, 2) - 1

    # - Layer generation
    fl0 = FFRateEulerJax(w_in=w_in, bias=0, noise_std=0.1, tau=20, dt=1,)

    # - Define simple input and target
    ts_input = TSContinuous([0, 1, 2, 3], [0, 1, 0, 0])
    ts_target = TSContinuous(
        [0, 1, 2, 3], np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]).T
    )

    # - Initialise training
    l, g, output_fcn = fl0.train_output_target(
        ts_input, ts_target, is_first=True, debug_nans=True,
    )

    # - Test output function
    output_fcn()

    # - Perform intermediate training step
    fl0.train_output_target(ts_input, ts_target)

    # - Perform final training step
    fl0.train_output_target(ts_input, ts_target, is_last=True)

    # - test batched training
    inp_batch = np.repeat(np.expand_dims(ts_input.samples, 0), 2, axis=0)
    tgt_batch = np.repeat(np.expand_dims(ts_target.samples, 0), 2, axis=0)
    fl0.train_output_target(
        inp_batch, tgt_batch, is_first=True, debug_nans=True, batch_axis=0,
    )
