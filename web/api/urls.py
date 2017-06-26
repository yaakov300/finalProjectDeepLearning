from django.conf.urls import url, include
from rest_framework.urlpatterns import format_suffix_patterns
from .views import *

urlpatterns = {
    # url(r'^bucketlists/$', CreateView.as_view(), name="create"),
    # url(r'^networks/$', CreateNetrorkView.as_view(), name="createNetwork"),
    # url(r'^signup/$', SignUpView.as_view(), name='signup'),
    url(r'^hello/', hello),
    url(r'^train/start/', train_network),
    url(r'^apply/test/', aplly_test),
}

urlpatterns = format_suffix_patterns(urlpatterns)