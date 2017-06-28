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
    # user = request.user
    if request.method == "POST" and request.is_ajax():
        data = request.body.decode('utf-8')
        network_dict = json.loads(data)
        start_training = model_app.add_network(network_dict['name'], network_dict)
        if start_training is not None:
            global_var.update_app_networks(model_app.networks)
            return HttpResponse(network_dict['name'] + " complete loading file \n started training wait for redirect "
                                                       "to dashboard page its will take few second", status=200)
        else:
            return HttpResponse(status=204)
    else:
        return HttpResponse(status=204)


@csrf_exempt
def continue_train_network(request):
    if request.method == "POST" and request.is_ajax():
        network_name = request.body
        net = model_app.get_network_by_name(network_name)
        if net is not None:
            net.continue_training()
            return {"status": True}
            global_var.update_app_networks(model_app.networks)
            return HttpResponse("ok", status=200)
        else:
            return HttpResponse(status=204)
    else:
        return HttpResponse(status=204)


@csrf_exempt
def testing_network(request):
    if request.method == "POST" and request.is_ajax():
        network_name = request.body
        result = model_app.testing_network(network_name)
        if result is not None:
            return HttpResponse(result, status=200)
        else:
            return HttpResponse(status=204)
    else:
        return HttpResponse(status=204)


@csrf_exempt
def stop_train_network(request):
    if request.method == "POST" and request.is_ajax():
        network_name = request.body
        net = model_app.get_network_by_name(network_name)
        if net is not None:
            is_stop = net.stop_training()

            global_var.update_app_networks(model_app.networks)
            return HttpResponse("ok", status=200)
        else:
            return HttpResponse(status=204)
    else:
        return HttpResponse(status=204)


def render_network_list():
    render_to_response("networks,html", {'networks': model_app.networks})


@csrf_exempt
def aplly_test(request):
    if request.method == "POST" and request.is_ajax():
        remove_temp()
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location = 'static/temp/test/')

        filename = fs.save(myfile.name, myfile)
        network_name = request.POST['network_name']
        test_network = model_app.get_network_by_name(network_name)
        result_use = test_network.running()
        uploaded_file_url = fs.url(filename)
        # print uploaded_file_url
        # return render(request, 'core/simple_upload.html', {
        #     'uploaded_file_url': uploaded_file_url
        # })

        jsonResponse = {'statu':'Success'}
        return JsonResponse(result_use)

    else:
        status = "Bad"
        return HttpResponse(" status: {} \n".format(status))


@csrf_exempt
def visualition_test(request):
    if request.method == "POST" and request.is_ajax():
        remove_temp()
        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location = 'static/temp/visual/')
        filename = fs.save(myfile.name, myfile)

        network_name = request.POST['network_name']
        layer_name = request.POST['layer_select']
        select_step = int(request.POST['select-step'])


        visual_network = model_app.get_network_by_name(network_name)
        result_visual = visual_network.visualising(layer_name = layer_name, return_steps = select_step)
        return render_to_response("_visual_galery.html", {'names_images': result_visual})
    else:
        status = "Bad"
        return HttpResponse(" status: {} \n".format(status))


def remove_temp():
    temp_opath = os.path.join(root_dir,'web/static/temp')
    directorys = ["test", "visual", "visual_gallery"]
    for directory in directorys:
        for file in os.listdir(os.path.join(temp_opath,directory)):
            if file.endswith('.jpg') or file.endswith('.png'):
                os.remove(os.path.join(temp_opath,directory,file))