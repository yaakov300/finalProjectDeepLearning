# api/serializers.py

from rest_framework import serializers
from .models import *
from django.conf import settings


class BucketlistSerializer(serializers.ModelSerializer):
    """Serializer to map the Model instance into JSON format."""

    class Meta:
        """Meta class to map serializer's fields with the model fields."""
        model = Bucketlist
        fields = ('id', 'name', 'date_created', 'date_modified')
        read_only_fields = ('date_created', 'date_modified')


class NetworklistSerializer(serializers.ModelSerializer):
    """Serializer to map the Model instance into JSON format."""

    class Meta:
        model = Network
        fields = ('id', 'name', 'path', 'date_created', 'date_modified')
        read_only_fields = ('date_created', 'date_modified')
