import tensorflow as tf
import dataset
import numpy as np
import matplotlib.pyplot as plt
import math
import yaml

#load config
config = yaml.safe_load(open("config.yml"))

#create graph
sess = tf.Session()
saver = tf.train.import_meta_graph(config['training']['model_path']+config['training']['model_name'])
saver.restore(sess, tf.train.latest_checkpoint(config['training']['model_path']+'./'))
graph = tf.get_default_graph()

#read image for test
y_pred = graph.get_tensor_by_name("y_pred:0")
test_path = config['training']['testeing_path']
img_size = 128
classes = config['training']['classes']
num_classes = len(classes)
num_channels = config['training']['num_channels']
img_size_flat = img_size * img_size * num_channels
test_images, test_ids, test_label = dataset.read_test_set(test_path, img_size,classes)
num_images = len(test_images)
print "len = {}".format(len(test_images))

x= graph.get_tensor_by_name("x:0")
x_batch = test_images.reshape(len(test_images), img_size_flat)

y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = test_label

print "y_test_images = {}".format(y_test_images)
y_test_images = np.zeros((len(test_images), 3))
feed_dict_testing = {x: x_batch, y_true: y_test_images}
feed_dict_visual = {x: x_batch[0], y_true: y_test_images[0]}

def test():
  pred = sess.run(y_pred, feed_dict=feed_dict_testing)
  msg = "animals = {0:6.5%}, cars = {1:6.5%}, people = {2:6.5%}"
  for i in range(num_images):
        print msg.format(pred[i][0],pred[i][1],pred[i][2])


def main ():
    test()


if __name__ == "__main__":
    main()
