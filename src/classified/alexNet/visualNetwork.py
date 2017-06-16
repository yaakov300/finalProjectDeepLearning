import cv2
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

def visualitzing():
    getActivations(graph.get_tensor_by_name("Conv2D:0"),feed_dict_testing)

def getActivations(layer,dict):
    units = sess.run(layer,feed_dict=dict)
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(3, figsize=(20,20))
    n_columns = 100
    n_rows = math.ceil(filters / n_columns) + 1
    # for i in range(filters):
    #     plt.subplot(n_rows, n_columns, i+1)
    #     plt.title('Filter ' + str(i))
    #     plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    #plt.s(n_rows, n_columns,1)
    plt.title('Filter ' + str(1))
    print "len image before resize = {}".format(len(units[0, :, :, 1]))
    image = cv2.resize(units[0, :, :, 1], (200, 200), 0, 0, cv2.INTER_LINEAR)
    print "len image before resize= {}".format(len(image))

    #plt.imshow(image)
    plt.imshow(image)#, interpolation="nearest", cmap="gray")
    plt.show()
    print "finish"



def main ():
    visualitzing()

if __name__ == "__main__":
    main()

