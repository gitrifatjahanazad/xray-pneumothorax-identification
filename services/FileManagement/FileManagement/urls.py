from django.contrib import admin
from django.urls import path
from FileUpload import urls
from django.conf.urls import url, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('FileUpload.urls')),
    path('api-auth/', include('rest_framework.urls')),
]
