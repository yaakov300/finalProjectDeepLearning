from CNN_layer import *
import src.classified.alexNet.dataset
import yaml
import json
import os

# region Define arguments

layers = {}
network = {}
outpot_folder = None
outpot_model = None
base_dir = os.path.dirname(__file__)

# open network config
with open(os.path.join(base_dir, 'alexnet.json')) as data_file:
    network_config = json.load(data_file)
    data_file.close()

# load global config
config = yaml.safe_load(open("config.yml"))

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = config['training']['num_channels']
# class info
classes = config['training']['classes']
num_classes = len(classes)
batch_size = config['training']['batch_size']
# files path
train_path = config['training']['training_path']
output_training_path = config['training']['output_training_path']

# image dimensions
img_size = network_config["input"]["img_size"]
# validation split
validation_size = network_config["runnig_config"]["validation_size"]
# how long to wait after validation loss stops improving before terminating training
early_stopping = network_config["runnig_config"][
    "early_stopping"]  # use None if you don't want to implement early stoping
# -------------------read and create dataset-----------------
# read
data = src.classified.alexNet.dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

img_size_flat = img_size * img_size * num_channels
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name=config["tensor_name"]["input_x"])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name=config["tensor_name"]["input_y_true"])

total_iterations = 0


# endregion

# region network_layer
# -------------------Network layer functions-----------------
def conv_layer(layer_parameters, prev_layer):
    layer, weights = new_conv_layer(input=prev_layer[1], filter_size=layer_parameters["conv_filter_size"],
                                    stride_size=layer_parameters["conv_stride_size"],
                                    num_filters=layer_parameters["num_filters"],
                                    num_input_channels=layer_parameters["num_input_channels"],
                                    layer_name=layer_parameters["name"],
                                    use_pooling=layer_parameters["use_pooling"],
                                    pooling_filter_size=layer_parameters["pooling_filter_size"],
                                    pooling_strides_size=layer_parameters["pooling_strides_size"])
    print "init conv\n"
    return [1, layer, weights]


def flatten_layer(layer_parameters, prev_layer):
    layer, num_features = new_flatten_layer(prev_layer[1])
    print "init flatten layer\n"
    return [2, layer, num_features, None]


def fc_layer(layer_parameters, prev_layer):
    outputs = layer_parameters["num_outputs"] if prev_layer[3] == None else prev_layer[3]
    layer = new_fc_layer(input=prev_layer[1],
                         num_inputs=prev_layer[2],
                         num_outputs=outputs,
                         use_relu=layer_parameters["use_relu"])
    print "init fc layer\n"
    return [3, layer, outputs, None]


def first_layer(layer_parameters, prev_layer):
    # input
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

    print "init first conv layer\n"
    return conv_layer(layer_parameters, [1, x_image])


def last_layer(layer_parameters, prev_layer):
    print "init last fc layer\n"
    return fc_layer(layer_parameters, [3, prev_layer[1], prev_layer[2], num_classes])


def init_layers():
    option = {0: first_layer,
              1: conv_layer,
              2: flatten_layer,
              3: fc_layer,
              4: last_layer
              }
    count_layers = 0
    prev_layer = None
    for l in network_config["layers"]:
        prev_layer = option[l["type"]](l["structure"], prev_layer)
        layers[count_layers] = prev_layer
        count_layers += 1
        print count_layers
    print "finish\n"


# endregion

# region save network
def create_folder():
    global outpot_folder
    outpot_folder = config["training"]["output_training_path"]
    if not os.path.exists(outpot_folder):
        os.makedirs(outpot_folder)
    outpot_folder = os.path.join(outpot_folder, network_config["name"])
    if not os.path.exists(outpot_folder):
        os.makedirs(outpot_folder)
    global outpot_model
    outpot_model = os.path.join(outpot_folder, "model")
    if not os.path.exists(outpot_model):
        os.makedirs(outpot_model)
    outpot_model = os.path.join(outpot_model, "modle")


def save_network_config_file():
    file_config = os.path.join(outpot_folder, "config.json")


def update_network_messege_file():


#endregion

# region training the network
def run_network():
    create_folder()
    output_layer = layers[len(layers) - 1][1]

    y_pred = tf.nn.softmax(output_layer, name=config["tensor_name"]["y_pred"])
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    # desirable outout
    y_true_cls = tf.argmax(y_true, dimension=1)

    # -------------------train the network network-----------------
    session = tf.Session()

    # cost function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    # optimizer gradient descent optimizier
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    # for print accuracy
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name=config["tensor_name"]["accuracy"])

    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
        # Calculate the accuracy on the training-set.
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
        msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"

        print(msg.format(epoch + 1, acc, val_acc, val_loss))

    def save_session(sess, i):
        saver.save(sess, outpot_model)

    def optimize(num_iterations):
        global total_iterations

        num_iterations_for_saving = 1

        print "data.train.num_examples = {}".format(data.train.num_examples)
        for i in range(total_iterations,
                       total_iterations + num_iterations):
            print i
            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)

            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

            x_batch = x_batch.reshape(batch_size, img_size_flat)
            x_valid_batch = x_valid_batch.reshape(batch_size, img_size_flat)
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            feed_dict_validate = {x: x_valid_batch,
                                  y_true: y_valid_batch}

            session.run(optimizer, feed_dict=feed_dict_train)

            if i % int(10) == 0:
                val_loss = session.run(cost, feed_dict=feed_dict_validate)
                epoch = int(i / 10)
                print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
                print "num_iterations_for_saving = {}".format(num_iterations_for_saving)

                if num_iterations_for_saving == network_config["runnig_config"]["num_iterations_for_saving"]:
                    num_iterations_for_saving = 0
                    print"save_session"
                    save_session(session, i)

                num_iterations_for_saving = 1 + num_iterations_for_saving

        total_iterations += num_iterations

    optimize(num_iterations=network_config["runnig_config"]["num_iterations"])


# endregion

init_layers()
run_network()
