# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import importlib

from django.http import HttpResponse
from django.shortcuts import render, render_to_response

from rest_framework import generics
from .serializers import *

from .models import *
from django.contrib.auth.forms import UserCreationForm
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import os, sys, inspect

# tmp = os.path.join(settings.CLASSIFIED_SETTING['app']['root'], settings.CLASSIFIED_SETTING['app']['src'])
tmp = settings.CLASSIFIED_SETTING['app']['root']
print tmp
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], tmp)))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import src.classified.cnn



def import_app_src(src, name):
    filename = "directory/module.py"
    directory = src
    module_name = name
    module_name = os.path.splitext(module_name)[0]

    path = list(sys.path)
    sys.path.insert(0, directory)
    try:
        module = __import__(module_name)
    finally:
        sys.path[:] = path


class CreateView(generics.ListCreateAPIView):
    """This class defines the create behavior of our rest api."""
    queryset = Bucketlist.objects.all()
    serializer_class = BucketlistSerializer

    def perform_create(self, serializer):
        """Save the post data when creating a new bucketlist."""
        serializer.save()


# class CreateNetrorkView(generics.ListCreateAPIView):
#     """This class defines the create behavior of our rest api."""
#     queryset = Network.objects.all()
#     serializer_class = NetworklistSerializer
#
#     def perform_create(self, serializer):
#         """Save the post data when creating a new bucketlist."""
#         serializer.save()


class SignUpView(CreateView):
    template_name = 'signup.html'
    form_class = UserCreationForm


def hello(request):
    return HttpResponse('Hello World!')


@csrf_exempt
def networkList(request):
    print request if request == None else "shimon"
    # user = request.user
    if request.method == "POST" and request.is_ajax():
        name = request.POST['name']
        network = Network(name)
        status = "Good"
        return render_to_response('hello.html', {'variable': network.name})
    else:
        status = "Bad"
        return render_to_response('hello.html', {'variable': status})
