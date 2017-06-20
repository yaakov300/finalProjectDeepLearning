# website/urls.py
from django.conf.urls import url
from website import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    url(r'^$', views.HomePageView.as_view(), name="deshboard"),
    url(r'^apply/$', views.ApplayPageView.as_view(), name="apply"),
    url(r'^train/$', views.TrainingPageView.as_view(),name="train"),
    # url(r'static/')


]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += script(settings.SCRIPT_URL, document_root=settings.SCRIPT_ROOT)
