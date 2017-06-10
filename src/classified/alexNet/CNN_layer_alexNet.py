import tensorflow as tf


def new_conv_layer(input, filter_size, num_filters, num_input_channels,
                   use_pooling=True, pooling_filter_size= "NONE", pooling_strides_size= "NONE"):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    stride = [1, 4, 4, 1]

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=stride,
                         padding='VALID')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, pooling_filter_size, pooling_filter_size, 1],
                               strides=[1, pooling_strides_size, pooling_strides_size, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)
    return layer,weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05)) # stddev -> The standard deviation

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))