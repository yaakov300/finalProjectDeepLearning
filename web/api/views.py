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

        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location = 'static/temp/test/')

        filename = fs.save(myfile.name, myfile)
        network_name = request.POST['network_name']
        test_network = model_app.get_network_by_name(network_name)
        result_use = test_network.running()
        # print result_use['cars']
        uploaded_file_url = fs.url(filename)
        print uploaded_file_url
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

        myfile = request.FILES['myfile']
        fs = FileSystemStorage(location = 'static/temp/visual/')
        filename = fs.save(myfile.name, myfile)

        network_name = request.POST['network_name']
        layer_name = request.POST['layer_select']
        select_step = int(request.POST['select-step'])

        # print "network_name: {0}, layer_name: {1}, select_step {2}".format(network_name, layer_name, select_step)
        visual_network = model_app.get_network_by_name(network_name)
        result_visual = visual_network.visualising(layer_name = layer_name, return_steps = select_step)
        print "############# len: {} #############3".format(len(result_visual))

        return HttpResponse([1,2])


        # jsonResponse = {'statu':'Success'}
        # result_use = jsonResponse
        # return JsonResponse(result_use)

    else:
        status = "Bad"
        return HttpResponse(" status: {} \n".format(status))