import json
import os
import cPickle as pickle
import yaml
from time import gmtime, strftime, time
import src.classified.dataset
from CNN_layer import *
from django.conf import settings

# region Define arguments



root_dir = settings.CLASSIFIED_SETTING['app']['root']
base_dir = os.path.dirname(__file__)

count_conv_layer = 0
count_flatten_layer = 0
count_fc_layer = 0
num_of_complete_iterations = 0


# endregion

def create_new_cnn(network_config, mode, import_layers=None):
    network_dict = {"layers": {}, "progress": {"status": None,
                                               "log": []}, }

    layers_name = []
    outpot_folder = None
    outpot_model = None
    global count_conv_layer
    count_conv_layer = 0
    global count_flatten_layer
    count_flatten_layer = 0
    global count_fc_layer
    count_fc_layer = 0
    # load global config
    with open(os.path.join(base_dir, "config.yml"), 'r') as stream:
        config = yaml.load(stream)

    # Number of color channels for the images: 1 channel for gray-scale.
    num_channels = config['training']['num_channels']
    batch_size = config['training']['batch_size']
    # files path
    train_path = os.path.join(root_dir, config['training']['training_path'])
    output_training_path = os.path.join(root_dir, config['training']['output_training_path'])

    # image dimensions
    img_size = network_config["input"]["img_size"] if mode == 0 else network_config["network"]["img_size"]
    # validation split
    validation_size = network_config["runnig_config"]["validation_size"] if mode == 0 else network_config["network"][
        "validation_size"]
    # how long to wait after validation loss stops improving before terminating training
    # early_stopping = network_config["runnig_config"][
    #     "early_stopping"]  # use None if you don't want to implement early stoping
    total_iterations = network_config["runnig_config"]["num_iterations"] if mode == 0 else network_config['network'][
        'number_of_iteration']
    # class info
    classes = network_config['input']['classes'] if mode == 0 else network_config['network']['classes']
    num_classes = len(classes)
    # -------------------read and create dataset-----------------
    # read
    data = src.classified.dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

    img_size_flat = img_size * img_size * num_channels
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name=config["tensor_name"]["input_x"])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name=config["tensor_name"]["input_y_true"])
    network_dict["progress"]["status"] = {"last_modified": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                                          "num_of_complete_iterations": 0,
                                          "state": 0}

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
                                        pooling_strides_size=layer_parameters["pooling_strides_size"],
                                        padding_filter_size=layer_parameters["padding_filter_size"],
                                        padding_strides_size=layer_parameters["padding_strides_size"])
        global count_conv_layer
        count_conv_layer = count_conv_layer + 1
        print "init conv\n"
        return [1, layer, weights]

    def flatten_layer(layer_parameters, prev_layer):
        layer, num_features = new_flatten_layer(prev_layer[1], layer_name=layer_parameters["name"])
        global count_flatten_layer
        count_flatten_layer = count_flatten_layer + 1
        print "init flatten layer\n"
        return [2, layer, num_features, None]

    def fc_layer(layer_parameters, prev_layer):
        outputs = layer_parameters["num_outputs"] if prev_layer[3] == None else prev_layer[3]
        layer = new_fc_layer(input=prev_layer[1],
                             num_inputs=prev_layer[2],
                             num_outputs=outputs,
                             use_relu=layer_parameters["use_relu"], layer_name=layer_parameters["name"])
        print "init fc layer\n"
        global count_fc_layer
        count_fc_layer = count_fc_layer + 1
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
            network_dict["layers"][count_layers] = prev_layer
            layers_name.append(l["structure"]["name"])
            count_layers += 1
            print count_layers
        print "finish\n"

    def load_last_model_training():
        global num_of_complete_iterations, outpot_folder
        lock = os.path.join(outpot_folder, "lock.txt")
        file_progress = os.path.join(outpot_folder, "progress.p")
        while os.path.exists(lock):
            time.sleep(1)
        if os.path.exists(file_progress):
            with open(file_progress, 'rb') as f:
                p = pickle.load(f)
                num_of_complete_iterations = p["status"]["num_of_complete_iterations"]

    # endregion

    # region save network
    def create_folder():
        global outpot_folder
        outpot_folder = os.path.join(root_dir, config["training"]["output_training_path"])
        if not os.path.exists(outpot_folder):
            os.makedirs(outpot_folder)
        outpot_folder = os.path.join(outpot_folder,
                                     network_config["name"] if mode == 0 else network_config["network"]["name"])
        if not os.path.exists(outpot_folder):
            os.makedirs(outpot_folder)
        global outpot_model
        outpot_model = os.path.join(outpot_folder, "model")
        if not os.path.exists(outpot_model):
            os.makedirs(outpot_model)
        outpot_model = os.path.join(outpot_model, "model")

    def save_network_config_file():
        global outpot_folder
        print "&&& " + outpot_folder
        file_config = os.path.join(outpot_folder, "config.yml")
        print "%%%%    " + file_config
        file_data = dict(network=dict(
            name=network_config["name"] if mode == 0 else network_config["network"]["name"],
            num_examples=data.train.num_examples,
            img_size=img_size,
            number_of_conv_layer=count_conv_layer,
            number_of_flatten_layer=count_fc_layer,
            number_of_fc_layer=count_flatten_layer,
            number_of_iteration=total_iterations,
            name_of_layer=layers_name,
            classes=classes,
            validation_size=validation_size,
        ))
        with open(file_config, 'w') as outfile:
            yaml.dump(file_data, outfile, default_flow_style=False)
            outfile.close()

    def update_network_progress_file():
        global outpot_folder
        file_progress = os.path.join(outpot_folder, "progress.p")
        lock = os.path.join(outpot_folder, "lock.txt")

        while os.path.exists(lock):
            time.sleep(1)

        with open(file_progress, 'wb') as fp:
            open(lock, 'w')
            pickle.dump(network_dict["progress"], fp)
            fp.close()
            os.remove(lock)

    # endregion

    # region training the network
    def run_network():
        session = tf.Session()

        if mode == 0:
            saver = tf.train.Saver()
            output_layer = network_dict['layers'][len(network_dict["layers"]) - 1][1]
        elif mode == 1:
            global outpot_model
            saver = tf.train.import_meta_graph(outpot_model + '.meta')
            saver.restore(session, tf.train.latest_checkpoint(
                os.path.join(outpot_model[:outpot_model.find('/model')], "model", './')))
            graph = tf.get_default_graph()
            last_layer_name = network_config['network']['name_of_layer'][-1]
            output_layer = graph.get_tensor_by_name("{}:0".format(last_layer_name))

        y_pred = tf.nn.softmax(output_layer, name=config["tensor_name"]["y_pred"])
        y_pred_cls = tf.argmax(y_pred, dimension=1)

        # desirable outout
        y_true_cls = tf.argmax(y_true, dimension=1)

        # -------------------train the network network-----------------


        # cost function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y_true)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = None
        if mode == 0:
            # optimizer gradient descent optimizier
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, name="first_optimizer").minimize(cost)
        elif mode == 1:
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
            # with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
            # optimizer = tf.get_variable("first_optimizer")#tf.train.AdamOptimizer(learning_rate=1e-4,name="first_optimizer").minimize(cost)
            # assert tf.get_variable_scope().reuse == True

        # for print accuracy
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name=config["tensor_name"]["accuracy"])

        session.run(tf.global_variables_initializer())

        def save_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
            # msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"

            # print(msg.format(epoch + 1, acc, val_acc, val_loss))
            network_dict["progress"]["log"].append(
                {"training_accuracy": acc, "validation_accuracy": val_acc, "validation_loss": val_loss})

        def save_session(sess, complete_iterations):
            global outpot_folder, num_of_complete_iterations, outpot_model
            saver.save(sess, outpot_model)  # , global_step=complete_iterations)
            network_dict["progress"]["status"]["last_modified"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            network_dict["progress"]["status"]["num_of_complete_iterations"] = complete_iterations
            num_of_complete_iterations = complete_iterations
            stop = os.path.join(outpot_folder, "stop.p")
            if os.path.exists(stop):
                network_dict["progress"]["status"]["state"] = 2
            elif complete_iterations < total_iterations:
                network_dict["progress"]["status"]["state"] = 1
            else:
                network_dict["progress"]["status"]["state"] = 3
            update_network_progress_file()
            return network_dict["progress"]["status"]["state"] == 2

        def optimize(num_iterations):
            num_iterations_for_saving = 1

            # print "data.train.num_examples = {}".format(data.train.num_examples)
            for i in xrange(num_of_complete_iterations, num_iterations):
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
                    save_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
                    # print "num_iterations_for_saving = {}".format(num_iterations_for_saving)

                    if num_iterations_for_saving == (network_config["runnig_config"][
                                                         "num_iterations_for_saving"] if mode == 0 else
                                                     network_config["network"][
                                                         "num_iterations_for_saving"]):
                        num_iterations_for_saving = 0
                        # print"save_session"
                        if save_session(session, i):
                            return
                    num_iterations_for_saving = 1 + num_iterations_for_saving
            save_session(session, total_iterations)

        optimize(num_iterations=network_config["runnig_config"]["num_iterations"] if mode == 0 else
        network_config["network"]["number_of_iteration"])

    # endregion
    create_folder()
    if mode == 0:
        init_layers()
        save_network_config_file()
        update_network_progress_file()
    elif mode == 1:
        load_last_model_training()

    run_network()

# open network config
# with open(os.path.join(base_dir, 'alexnet.json')) as data_file:
#     json_file = json.load(data_file)
#     data_file.close()
#     addCnn(json_file)
