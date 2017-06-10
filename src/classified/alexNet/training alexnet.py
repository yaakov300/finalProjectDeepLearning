from CNN_layer_alexNet import *
import dataset

#-------------------Define arguments-----------------

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3
# image dimensions
img_size = 128
# class info
classes = ['animals', 'people', 'cars']
num_classes = len(classes)
#TODO repalce batch size.
batch_size = 16
# validation split
validation_size = .2
# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping
#files path
train_path='training_data'
test_path='testing_data'


#-------------------read and create dataset-----------------
#read
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, img_size,classes)

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)


# input
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size ,num_channels])

# desirable outout
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


#-------------------Define network-----------------
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

#layer_conv3
#padding layer 2
padded_conv_layer_2 = tf.pad(layer_conv2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
num_filters_layer3 = 384
num_input_channels_layer3 = num_filters_layer2
conv_filter_size_layer3 = 3
conv_stride_size_layer3= 1

layer_conv3, weights_conv3 = new_conv_layer(input=padded_conv_layer_2, filter_size=conv_filter_size_layer3,
                                            stride_size=conv_stride_size_layer3,
                                            num_filters=num_filters_layer3,
                                            num_input_channels=num_input_channels_layer3,
                                            use_pooling=False)

#layer_conv4
#padding layer 3
padded_conv_layer_3 = tf.pad(layer_conv3, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
num_filters_layer4 = 384
num_input_channels_layer4 = num_filters_layer3
conv_filter_size_layer4 = 3
conv_stride_size_layer4= 1

layer_conv4, weights_conv4 = new_conv_layer(input=padded_conv_layer_3, filter_size=conv_filter_size_layer4,
                                            stride_size=conv_stride_size_layer4,
                                            num_filters=num_filters_layer4,
                                            num_input_channels=num_input_channels_layer4,
                                            use_pooling=False)

#layer_conv5
#padding layer 4
padded_conv_layer_4 = tf.pad(layer_conv4, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
num_filters_layer5 = 256
num_input_channels_layer5 = num_filters_layer4
conv_filter_size_layer5 = 3
conv_stride_size_layer5= 1
pooling_filter_size_layer5 = 3
pooling_strides_size_layer5 = 2

layer_conv5, weights_conv5 = new_conv_layer(input=padded_conv_layer_4, filter_size=conv_filter_size_layer5,
                                            stride_size=conv_stride_size_layer5,
                                            num_filters=num_filters_layer5,
                                            num_input_channels=num_input_channels_layer5,
                                            use_pooling=True,
                                            pooling_filter_size=pooling_filter_size_layer5,
                                            pooling_strides_size=pooling_strides_size_layer5)


#flat conv5
layer_flat, num_features = flatten_layer(layer_conv5)
# fully conectted 1
fc_size1 = 4096
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size1,
                         use_relu=True)
# fully conectted 1
fc_size2 = 4096
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size1,
                         num_outputs=fc_size2,
                         use_relu=True)
# fully conectted 1
num_classes = 3
layer_fc3 = new_fc_layer(input=layer_fc2,
                         num_inputs=fc_size2,
                         num_outputs=num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc3)
y_pred_cls = tf.argmax(y_pred, dimension=1)


#-------------------train the network network-----------------
session = tf.Session()
#sess.run(tf.global_variables_initializer())

# cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# optimizer gradient descent optimizier
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())
train_batch_size = batch_size

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

print "x: {x}".format(x = sess.run(tf.shape(x)))
print "x_image: {x_image}".format(x_image = sess.run(tf.shape(x_image)))
print "weights_conv1: {weights_conv1}".format(weights_conv1= sess.run(tf.shape(weights_conv1)))
print "layer_conv1: {layer_conv1}".format(layer_conv1 = sess.run(tf.shape(layer_conv1)))
print "weights_conv2: {weights_conv2}".format(weights_conv2= sess.run(tf.shape(weights_conv2)))
print "layer_conv2: {layer_conv2}".format(layer_conv2 = sess.run(tf.shape(layer_conv2)))
print "weights_conv3: {weights_conv3}".format(weights_conv3= sess.run(tf.shape(weights_conv3)))
print "layer_conv3: {layer_conv3}".format(layer_conv3 = sess.run(tf.shape(layer_conv3)))
print "weights_conv4: {weights_conv4}".format(weights_conv4= sess.run(tf.shape(weights_conv4)))
print "layer_conv4: {layer_conv4}".format(layer_conv4 = sess.run(tf.shape(layer_conv4)))
print "weights_conv5: {weights_conv5}".format(weights_conv5= sess.run(tf.shape(weights_conv5)))
print "layer_conv5: {layer_conv5}".format(layer_conv5 = sess.run(tf.shape(layer_conv5)))
print "layer_flat: {layer_flat}".format(layer_flat = sess.run(tf.shape(layer_flat)))
print "layer_fc1: {layer_fc1}".format(layer_fc1 = sess.run(tf.shape(layer_fc1)))
print "layer_fc2: {layer_fc2}".format(layer_fc2 = sess.run(tf.shape(layer_fc2)))
print "layer_fc3: {layer_fc3}".format(layer_fc3 = sess.run(tf.shape(layer_fc3)))
print "y_pred: {y_pred}".format(y_pred = sess.run(tf.shape(y_pred)))
print "y_pred_cls: {y_pred_cls}".format(y_pred_cls = sess.run(tf.shape(y_pred_cls)))


# input = tf.placeholder(tf.float32, [None, 28, 28, 3])
# padded_input = tf.pad(input, [[0, 0], [2, 2], [1, 1], [0, 0]], "CONSTANT")
#
# filter = tf.placeholder(tf.float32, [5, 5, 3, 16])
# output = tf.nn.conv2d(padded_input, filter, strides=[1, 1, 1, 1], padding="VALID")