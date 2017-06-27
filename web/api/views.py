# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import importlib
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.http import JsonResponse
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
    print request if request == None else "shimon"
    # user = request.user
    if request.method == "POST" and request.is_ajax():
        data = request.body.decode('utf-8')
        network_dict = json.loads(data)
        model_app.add_network(network_dict['name'], network_dict)

        return HttpResponse(network_dict['name'] + " status:\n")
    else:
        status = "Bad"
        return render_to_response('hello.html', {'variable': status})


def render_network_list():
    render_to_response("networks,html", {'networks': model_app.networks})


@csrf_exempt
def aplly_test(request):
    if request.method == "POST" and request.is_ajax():

        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location = 'static/temp/test/')

        filename = fs.save(myfile.name, myfile)
        network_name = request.POST['network_name']
        test_network = model_app.get_network_by_name(network_name)
        result_use = test_network.running()
        # print result_use['cars']
        # uploaded_file_url = fs.url(filename)
        # return render(request, 'core/simple_upload.html', {
        #     'uploaded_file_url': uploaded_file_url
        # })

        jsonResponse = {'statu':'Success'}
        return JsonResponse(result_use)

    else:
        status = "Bad"
        return HttpResponse(" status: {} \n".format(status))