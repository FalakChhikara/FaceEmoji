from django.urls import path

from . import views

app_name = 'facetoemoji'

urlpatterns = [

    path(r'', views.homepage, name='homepage'),
    path(r'/webcam', views.webcam, name='webcam'),

]