# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os, sys, inspect
import yaml
from django.conf import settings
import pickle
from threading import Thread

root_dir = settings.CLASSIFIED_SETTING['app']['root']
base_dir = os.path.dirname(__file__)
network_dir = os.path.join(root_dir, settings.CLASSIFIED_SETTING['app']['networks_dir'])

# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], root_dir)))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from src.classified import testing_network, visualising_network, cnn

# import src.classified.cnn

status = {0: "loading",
          1: "training",
          2: "pause",
          3: "complete"}


# class TrainingThread(Thread):
#     def __init__(self, network_config):
#         Thread.__init__(self)
#         self.network_config = network_config
#
#     def run(self):
#         cnn.create_new_cnn(self.network_config)



class Network:
    def __init__(self, name, network_config=None):
        self.name = name
        self.status = 0
        self.iterations_completed = 0
        self.training_accuracy = 0
        self.validation_accuracy = 0
        self.validation_loss = 0
        self.config = None
        self.progress = None
        self.last_modified = None
        self.training_thread = None
        if network_config is not None:
            self.start_training(network_config)

    def start_training(self, network_config):
        if self.training_thread is None:
            self.training_thread = Thread(target=cnn.create_new_cnn, args=(network_config, 0,),
                                          name=network_config['name'])
            self.training_thread.start()
            print r"++ start training {self.name} ++"

    def load_network_config(self):
        config_file = os.path.join(network_dir, self.name, "config.yml")
        if self.config is None and os.path.exists(config_file):
            with open(os.path.join(network_dir, self.name, "config.yml"), 'r') as stream:
                self.config = yaml.load(stream)
                print r"++ load config {self.name} ++"

    def update_network_progress(self):
        progress_file = os.path.join(network_dir, self.name, "progress.p")
        if os.path.exists(progress_file):
            with open(progress_file, 'rb') as f:
                self.progress = pickle.load(f)
                print r"++ load progress {self.name} ++"
            if self.progress is not None:
                p = self.progress
                self.status = p["status"]["state"] if self.iterations_completed == self.config["network"][
                    "number_of_iteration"] or self.training_thread is not None else 2
        self.iterations_completed = p["status"]["num_of_complete_iterations"]
        self.last_modified = p["status"]["last_modified"]
        self.training_accuracy = p["log"][0]["training_accuracy"]
        self.validation_accuracy = p["log"][0]["validation_accuracy"]
        self.validation_loss = p["log"][0]["validation_loss"]
        print r"++ load s\progress log {self.name} ++"

    def get_status(self):
        return status[self.status]

    def testing(self, model_name=None):
        if self.status is 0:
            return "network still loading state"
        net = {
            "model": {
                "configPath": self.name,
                "path": os.path.join(self.name, "model"),
                "name": model_name if not None else os.path.join(self.name, "model/model.meta")
            },
            "testing": True
        }
        return testing_network.testing_network(self.config, net)

    def running(self, img, model_name=None):
        if self.status is 0:
            return "network still loading state"
        net = {
            "model": {
                "configPath": self.name,
                "path": os.path.join(self.name, "model"),
                "name": model_name if not None else os.path.join(self.name, "model/model.meta")
            },
            "testing": False,
            "img": img
        }
        return testing_network.testing_network(self.config, net)

    def visualising(self, layer_name, f_index, f_step, return_steps):
        if self.status is 0:
            return "network still loading state"
        net = {
            "model": {
                "configPath": self.name,
                "path": os.path.join(self.name, "model"),
                "name": "alexNet_model1370.meta"
            },
            "data": {
                "imgSize": self.config['img_size'],
                "numClasses": len(self.config['classes'])
            },
            "layers": {
                "Names": layer_name
            },
            "filter": {
                "indexFilter": f_index,
                "stepFilter": f_step
            },
            "return": {
                "steps": return_steps
            }
        }

        return visualising_network.visual_network(self.config, net)


class Networks:
    def __init__(self):
        self.networks = []

    def add_network(self, name, network_config=None):
        net = Network(name, network_config)
        net.load_network_config()
        net.update_network_progress()
        self.networks.append(net)

    def get_network_by_name(self, name):
        for net in self.networks:
            if net.name == name:
                return net
        return None

    def get_networks_tree(self):
        networks_tree = []
        for net in self.networks:
            networks_tree.append({'name': net.name, 'classes': net.config["network"]["classes"],
                                  'conv_layers': net.config["network"]["name_of_layer"]})
        return networks_tree

    def load_existing_networks(self):
        networks_name = [item for item in os.listdir(network_dir) if os.path.isdir(os.path.join(network_dir, item))]
        for net in networks_name:
            self.add_network(net)
