from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from rest_framework.schemas import get_schema_view
from FileUpload.views import FileUploadView

urlpatterns = [
   path('', FileUploadView.as_view()),
]

urlpatterns = format_suffix_patterns(urlpatterns)