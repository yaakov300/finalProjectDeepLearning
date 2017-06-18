# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# website/views.py
from django.shortcuts import render
from django.views.generic import TemplateView
from model.networks import Network
from model.classes import Classes
import global_var

count = global_var


# Create your views here.

# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        networks = Network()
        print networks.name
        context_dict = {'networks': [networks,networks],'deshboardActive': 'active'}
        return render(request, 'deshboard.html', context_dict)

class ApplayPageView(TemplateView):
    def get(self, request, **kwargs):
        classe = Classes()
        context_dict = {'classes': classe, 'applyActive':'active'}
        return render(request, 'apply.html', context_dict)

class TrainingPageView(TemplateView):
    def get(self, request, **kwargs):
        context_dict = {'trainingActive':'active'}
        return render(request, 'training.html', context_dict)



# class DeshboardPageView(TemplateView):
#     def get(self, request, **kwargs):
#         return render(request, 'deshboard.html', context=None)