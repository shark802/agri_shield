"""
API URL Configuration
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('health/', views.health, name='health'),
    path('status/', views.status, name='status'),
    path('detect/', views.detect, name='detect'),
    path('classify/', views.classify, name='classify'),
    path('forecast/', views.forecast, name='forecast'),
    path('forecast/quick/', views.forecast_quick, name='forecast_quick'),
    path('forecast/current/', views.forecast_current, name='forecast_current'),
    path('forecast/update/', views.forecast_update, name='forecast_update'),
    path('test_db/', views.test_db, name='test_db'),
]

