import tensorflow as tf
import numpy as np


def conv2d(x, output_dim, kernel_size, stride_size, data_format, initializer, activation=None, padding="VALID", name="conv2d"):
    """This method is designed for different data-format: NCHW and NHWC,
    return layer, weight-matrix and bias vector

    :param x: tf.Tensor, the input data
    :param output_dim: int
    :param kernel_size: list, indicates the kernel's size
    :param stride_size: list, indicates the size of stride
    :param data_format: str, choice {'NCHW', 'NHWC'}
    :param initializer: tf.nn.initializer
    :param activation: tf.nn.activation, indicates the activation which be used at this Conv-layer
    :param padding: list
    """

    with tf.get_variable_scope(name):
        if data_format == "NCHW":
            strides = [1, 1, stride_size[0], stride_size[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
        elif data_format == "NHWC":
            strides = [1, stride_size[0], stride_size[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        w = tf.get_variable("w", shape=kernel_shape, initializer=initializer)
        conv = tf.nn.conv2d(x, w, strides, padding, data_format=data_format)

        b = tf.get_variable("bias", shape=[output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)
    
    if activation is not None:
        return activation(out), w, b
    else:
        return out, w, b


def custom_dense(x, out_dim, activation, initializer, name="dense"):
    """Custom dense layer, return dense-layer, weight-matrix and bias.
    """

    w = tf.get_variable("w", shape=(x.get_shape()[1], out_dim), initializer=initializer)
    b = tf.get_variable("bias", shape=(out_dim,), initializer=tf.constant_initializer(0.0))
    
    out = tf.mat_mul(x, w) + b

    if activation is not None:
        return activation(out), w, b
    else:
        return out, w, b

