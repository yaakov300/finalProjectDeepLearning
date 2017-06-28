# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# website/views.py
from django.shortcuts import render
from django.views.generic import TemplateView

# from model.classes import Classes
from api import global_var

# Create your views here.
model_app = global_var.app_networks


# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        model_app.load_existing_networks()
        for net in model_app.networks:
            net.update_network_progress()
        context_dict = {'networks': model_app.networks, 'deshboardActive': 'active'}
        return render(request, 'deshboard.html', context_dict)


class ApplayPageView(TemplateView):
    def get(self, request, **kwargs):
        network_tree = model_app.get_networks_tree()
        context_dict = {'networks': network_tree, 'applyActive': 'active'}
        return render(request, 'apply.html', context_dict)


class TrainingPageView(TemplateView):
    def get(self, request, **kwargs):
        context_dict = {'trainingActive': 'active'}
        return render(request, 'training.html', context_dict)


class visualitionPageView(TemplateView):
    def get(self, request, **kwargs):
        network_tree = model_app.get_networks_tree()
        context_dict = {'networks': network_tree, 'visualitionActive': 'active'}
        return render(request, 'visualition.html', context_dict)
