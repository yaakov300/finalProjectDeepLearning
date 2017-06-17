import tensorflow as tf


def new_conv_layer(input, filter_size, stride_size, num_filters, num_input_channels,
                   layer_name, use_pooling=True, pooling_filter_size="NONE", pooling_strides_size="NONE",
                   padding_filter_size=None, padding_strides_size=None):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, stride_size, stride_size, 1],
                         padding="VALID",
                         name=layer_name)

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, pooling_filter_size, pooling_filter_size, 1],
                               strides=[1, pooling_strides_size, pooling_strides_size, 1],
                               padding='VALID',
                               name=layer_name + "_pool")

    if (padding_filter_size != None and padding_strides_size != None):
        tf.pad(layer, [[padding_strides_size, padding_strides_size], [padding_filter_size, padding_filter_size],
                       [padding_filter_size, padding_filter_size], [padding_strides_size, padding_strides_size]],
               "CONSTANT")

    layer = tf.nn.relu(layer)
    return layer, weights


def new_flatten_layer(layer):
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
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))  # stddev -> The standard deviation


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
