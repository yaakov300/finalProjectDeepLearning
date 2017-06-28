# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import importlib

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import render, render_to_response
from django.views.generic import CreateView

from rest_framework import generics
from .serializers import *

from .models import *
from django.contrib.auth.forms import UserCreationForm
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import os

import json

import global_var

root_dir = settings.CLASSIFIED_SETTING['app']['root']
base_dir = os.path.dirname(__file__)
network_dir = os.path.join(root_dir, settings.CLASSIFIED_SETTING['app']['networks_dir'])

model_app = global_var.app_networks


def hello(request):
    return HttpResponse('Hello World!')


@csrf_exempt
def train_network(request):
    # user = request.user
    if request.method == "POST" and request.is_ajax():
        data = request.body.decode('utf-8')
        network_dict = json.loads(data)
        if model_app.add_network(network_dict['name'], network_dict):
            global_var.update_app_networks(model_app.networks)
            return HttpResponse(network_dict['name'] + " complete loading file \n started training wait for redirect "
                                                       "to dashboard page its will take few second")
        else:
            return HttpResponse("error")
    else:
        return HttpResponse("error")


@csrf_exempt
def continue_train_network(request):
    if request.method == "POST" and request.is_ajax():
        network_name = request.body
        net = model_app.get_network_by_name(network_name)
        if net is not None:
            net.continue_training()
            return {"status": True}
            global_var.update_app_networks(model_app.networks)
        else:
            return {"status": False}
    else:
        return {"status": False}


@csrf_exempt
def testing_network(request):
    if request.method == "POST" and request.is_ajax():
        network_name = request.body
        net = model_app.get_network_by_name(network_name)
        if net is not None:
            net.testing()
            return {"status": True, }
        else:
            return {"status": False, }
    else:
        return {"status": False, }


@csrf_exempt
def stop_testing_network(request):
    if request.method == "POST" and request.is_ajax():
        network_name = request.body
        net = model_app.get_network_by_name(network_name)
        if net is not None:
            net.stop_training()
            global_var.update_app_networks(model_app.networks)
            return {"status": True, }
        else:
            return {"status": False, }
    else:
        return {"status": False, }


def render_network_list():
    render_to_response("networks,html", {'networks': model_app.networks})


@csrf_exempt
def aplly_test(request):
    if request.method == "POST" and request.is_ajax():
        status = "Success"
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        # uploaded_file_url = fs.url(filename)
        # return render(request, 'core/simple_upload.html', {
        #     'uploaded_file_url': uploaded_file_url
        # })
        return HttpResponse(" status:\n")
    else:
        status = "Bad"
        return HttpResponse(" status:\n")
