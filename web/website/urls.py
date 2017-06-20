# website/urls.py
from django.conf.urls import url
from website import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    url(r'^$', views.HomePageView.as_view()),
    url(r'^apply/$', views.ApplayPageView.as_view()),
    url(r'^train/$', views.TrainingPageView.as_view()),
    url(r'^visualition/$', views.visualitionPageView.as_view()),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += script(settings.SCRIPT_URL, document_root=settings.SCRIPT_ROOT)
