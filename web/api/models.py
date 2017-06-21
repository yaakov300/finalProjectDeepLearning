# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models


# class Network(models.Model):
#     name = models.CharField(max_length=255, blank=False, unique=True)
#     path = models.CharField(max_length=255, blank=False, unique=True)
#     date_created = models.DateTimeField(auto_now_add=True)
#     date_modified = models.DateTimeField(auto_now=True)
#
#     def __str__(self):
#         return "{}".format(self.name)#, self.path)
class Network():
    def __init__(self,name):
        self.name = name
        self.status = "loading"
        self.iterations_completed = 0
        self.training_accuracy = 0
        self.validation_accuracy = 0
        self.loss = 0

        def __str__(self):
            return "{}".format(self.name)



class Bucketlist(models.Model):
    """This class represents the bucketlist model."""
    name = models.CharField(max_length=255, blank=False, unique=True)
    date_created = models.DateTimeField(auto_now_add=True)
    date_modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        """Return a human readable representation of the model instance."""
        return "{}".format(self.name)
