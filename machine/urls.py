from django.conf.urls import url
from .views import machineLearning

app_name = 'machine'

urlpatterns = [
    url(r'^machine/$',machineLearning, name='machine'),
]