"""my_graduation_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ds_nre import views

urlpatterns = [
    path(r'admin/', admin.site.urls),

    path(r'index/', views.index),    #raw_html
    path(r'parameters_setting/', views.parameters_setting),
    path(r'pattern_filters/', views.pattern_filters),
    path(r'trainning_result/', views.trainning_result),
    path(r'model_prediction/', views.model_prediction),

    path(r'setting_parameters/', views.setting_parameters),  # setting_parameters

    path(r'load_parameters_file/', views.load_parameters_file),  # filter_pattern
    path(r'pattern_filter/', views.pattern_filter),
    path(r'filtered_pattern_checkout/', views.filtered_pattern_checkout),
    path(r'train/', views.train),

    path(r'load_model_name/', views.load_model_name),  # training_result
    path(r'train_result/', views.train_result),

    path(r'prediction/', views.prediction),  # prediction

]
