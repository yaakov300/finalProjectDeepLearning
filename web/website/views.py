# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# website/views.py
from django.shortcuts import render
from django.views.generic import TemplateView
from model.networks import Network

import global_var

count = global_var


# Create your views here.

# Create your views here.
class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        networks = Network()

        print networks.name
        context_dict = {'networks': [networks,networks]}
        return render(request, 'deshboard.html', context_dict)

class ApplayPageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'apply.html', context=None)


# class DeshboardPageView(TemplateView):
#     def get(self, request, **kwargs):
#         return render(request, 'deshboard.html', context=None)