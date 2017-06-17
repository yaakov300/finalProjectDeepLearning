# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# website/views.py
from django.shortcuts import render
from django.views.generic import TemplateView

import global_var

count = global_var


# Create your views here.

# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'deshboard.html', context=None)

class ModelTrainingPageView(TemplateView):
    def get(self, request, **kwargs):
        tmp = {'person_name': global_var.incress_training()}
        return render(request, 'modelTraining.html', context=tmp)

# class DeshboardPageView(TemplateView):
#     def get(self, request, **kwargs):
#         return render(request, 'deshboard.html', context=None)