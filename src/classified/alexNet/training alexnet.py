from CNN_layer_alexNet import *

# input
x = tf.placeholder(tf.float32, shape=[16, 49152], name='x')
x_image = tf.reshape(x, [-1, 128, 128 ,3])


#layer_conv1 + pooling
num_filters_layer1 = 96
num_input_channels_layer1 = 3
conv_filter_size_layer1 = 12
conv_stride_size_layer1= 2
pooling_filter_size_layer1 = 7
pooling_strides_size_layer1 = 2

layer_conv1, weights_conv1 = new_conv_layer(input=x_image, filter_size=conv_filter_size_layer1,
                                            stride_size=conv_stride_size_layer1,
                                            num_filters=num_filters_layer1,
                                            num_input_channels=num_input_channels_layer1,
                                            use_pooling=True,
                                            pooling_filter_size=pooling_filter_size_layer1,
                                            pooling_strides_size=pooling_strides_size_layer1)
#layer_conv2 + pooling
#padding layer 1
padded_conv_layer_1 = tf.pad(layer_conv1, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
num_filters_layer2 = 256
num_input_channels_layer2 = num_filters_layer1
conv_filter_size_layer2 = 5
conv_stride_size_layer2= 1
pooling_filter_size_layer2 = 3
pooling_strides_size_layer2 = 2

layer_conv2, weights_conv2 = new_conv_layer(input=padded_conv_layer_1, filter_size=conv_filter_size_layer2,
                                            stride_size=conv_stride_size_layer2,
                                            num_filters=num_filters_layer2,
                                            num_input_channels=num_input_channels_layer2,
                                            use_pooling=True,
                                            pooling_filter_size=pooling_filter_size_layer2,
                                            pooling_strides_size=pooling_strides_size_layer2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


print "x: {x}".format(x = sess.run(tf.shape(x)))
print "x_image: {x_image}".format(x_image = sess.run(tf.shape(x_image)))
print "weights_conv1: {weights_conv1}".format(weights_conv1= sess.run(tf.shape(weights_conv1)))
print "layer_conv1: {layer_conv1}".format(layer_conv1 = sess.run(tf.shape(layer_conv1)))
print "weights_conv2: {weights_conv2}".format(weights_conv2= sess.run(tf.shape(weights_conv2)))
print "layer_conv2: {layer_conv2}".format(layer_conv2 = sess.run(tf.shape(layer_conv2)))



# input = tf.placeholder(tf.float32, [None, 28, 28, 3])
# padded_input = tf.pad(input, [[0, 0], [2, 2], [1, 1], [0, 0]], "CONSTANT")
#
# filter = tf.placeholder(tf.float32, [5, 5, 3, 16])
# output = tf.nn.conv2d(padded_input, filter, strides=[1, 1, 1, 1], padding="VALID")