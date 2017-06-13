import tensorflow as tf
import dataset
import numpy as np
sess = tf.Session()
saver = tf.train.import_meta_graph('model_files/alexNet_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('model_files/./'))
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")

test_path='../testing_temp'
img_size = 128
classes = ['dogs','cats']
num_classes = len(classes)
num_channels = 3
img_size_flat = img_size * img_size * num_channels
test_images, test_ids, test_label = dataset.read_test_set(test_path, img_size,classes)
print "len = {}".format(len(test_images))
#print test_label
x= graph.get_tensor_by_name("x:0")
x_batch = test_images.reshape(len(test_images), img_size_flat)
#
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = test_label
#y_test_images = np.zeros((3, 3))
# y_test_images = [[0,0,1],[0,0,1],[0,0,1]]
#
# print "y_test_images = {}".format(y_test_images)
#
print "y_test_images = {}".format(y_test_images)
feed_dict_testing = {x: x_batch, y_true: y_test_images}
print(sess.run(y_pred, feed_dict=feed_dict_testing))


# print "-----------"
# print x_batch
# print test_ids
# print "-----------"