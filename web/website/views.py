# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# website/views.py
from django.shortcuts import render
from django.views.generic import TemplateView
from model.networks import Network
# from model.classes import Classes
import global_var

count = global_var


# Create your views here.

# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        status = {0: "load",
                  1: "training",
                  2: "complete"}
        networks1 = Network('alexNet', 95, 'training', 1600, 0.97, 0.05, 0.95)
        networks2 = Network('TEST', 81, 'training', 20, 0.67, 0.350, 0.62)
        context_dict = {'networks': [networks2, networks1], 'deshboardActive': 'active'}
        return render(request, 'deshboard.html', context_dict)


class ApplayPageView(TemplateView):
    def get(self, request, **kwargs):
        classes = Classes().classes_name_pred
        print classes
        context_dict = {'classes': classes, 'applyActive': 'active'}
        return render(request, 'apply.html', context_dict)


class TrainingPageView(TemplateView):
    def get(self, request, **kwargs):
        context_dict = {'trainingActive': 'active'}
        return render(request, 'training.html', context_dict)


class visualitionPageView(TemplateView):
    def get(self, request, **kwargs):
        context_dict = {'visualitionActive': 'active'}
        return render(request, 'visualition.html', context_dict)
        # visualition

        # class DeshboardPageView(TemplateView):
        #     def get(self, request, **kwargs):
        #         return render(request, 'deshboard.html', context=None)
