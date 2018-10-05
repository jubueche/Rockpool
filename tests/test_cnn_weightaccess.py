"""
Test weigh access and indexing for CNNWeight class
"""
import sys
import pytest
import numpy as np


def test_import():
    """
    Test import of the class
    """
    from NetworksPython.layers import CNNWeight


def test_raise_exception_on_incorrect_shape():
    """
    Test exception on size incompatibility
    """
    from NetworksPython.layers import CNNWeight

    W = CNNWeight(inShape=(200, 200))

    # Create an image
    myImg = np.random.rand(400, 400) > 0.999

    # Test indexing with entire image
    with pytest.raises(ValueError):
        W[myImg]


def test_raise_exception_on_undefined_shape():
    """
    Test exception on size incompatibility
    """
    from NetworksPython.layers import CNNWeight

    W = CNNWeight()

    # Create an image
    myImg = np.random.rand(400, 400) > 0.999
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    with pytest.raises(IndexError):
        W[myImgIndex]


def test_convolution_full_image():
    """
    Test convolution of full image
    """
    from NetworksPython.layers import CNNWeight

    W = CNNWeight(inShape=(400, 400))

    # Create an image
    myImg = np.random.rand(400, 400) > 0.999

    # Test indexing with entire image
    outConv = W[myImg]
    assert myImg.size == outConv.size


def test_convolutionl_nonzero_index():
    """
    Test convolution when the indexing is done by non-zero pixels
    """
    from NetworksPython.layers import CNNWeight

    W = CNNWeight(inShape=(400, 400))

    # Create an image
    myImg = np.random.rand(400, 400) > 0.999
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    outConv = W[myImgIndex]
    assert myImg.size == outConv.size


def test_data_format_channels_last():
    """
    Test indexing and output dimensions with channels last data format
    """
    from NetworksPython.layers import CNNWeight

    W = CNNWeight(
        inShape=(400, 400, 1),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_last",
    )

    # Create an image
    myImg = np.zeros((400, 400, 1))
    myImg[0, 5, 0] = 1  # One pixel in image active
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    outConv = W[myImgIndex]
    # Ensure size of output is as expected
    assert myImg.size * 3 == outConv.size
    # Ensure image dimensions are understood and maintained
    assert myImg.shape[:2] == W.outShape[:2]
    # Ensure convolution data is accurate
    outConv = outConv.reshape((400, 400, 3))
    assert outConv[0, 5, 0] != 0
    assert outConv[5, 0, 0] == 0


def test_data_format_channels_first():
    """
    Test indexing and output dimensions with channels last data format
    """
    from NetworksPython.layers import CNNWeight

    W = CNNWeight(
        inShape=(1, 400, 400),
        nKernels=3,
        kernel_size=(1, 1),
        mode="same",
        img_data_format="channels_first",
    )

    # Create an image
    myImg = np.zeros((1, 400, 400))
    myImg[0, 5, 0] = 1  # One pixel in image active
    myImgIndex = myImg.flatten().nonzero()[0]

    # Test indexing with entire image
    outConv = W[myImgIndex]
    # Ensure image dimensions are understood and maintained
    assert myImg.shape[-2:] == W.outShape[-2:]
    # Ensure size of output is as expected
    assert myImg.size * 3 == outConv.size
    # Ensure convolution data is accurate
    outConv = outConv.reshape((3, 400, 400))
    assert outConv[0, 5, 0] != 0
    assert outConv[0, 0, 5] == 0


def test_strides_on_convolution():
    """
    Test the convolution upon a custom stride specified by user
    """
    from NetworksPython.layers import CNNWeight

    W = CNNWeight(
        inShape=(1, 10, 10),
        nKernels=3,
        kernel_size=(2, 2),
        strides=(2, 5),
        mode="valid",
        img_data_format="channels_first",
    )

    assert W.outShape == (3, 5, 2)