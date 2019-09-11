from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework import generics, permissions, renderers
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.reverse import reverse
from .serializers import FileSerializer
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
from rest_framework import status
from PIL import Image
import json
import base64
from io import BytesIO

@permission_classes((permissions.AllowAny,))
class FileUploadView(APIView):
    parser_class = [FileUploadParser]
    
    def post(self, request, *args, **kwargs):
      file_serializer = FileSerializer(data=request.data)
      print(request.FILES.values)
      print(request.FILES.values())
      print(file_serializer.is_valid())
      

      buffered = BytesIO()
      image = Image.open(request.data["file"])
      image = image.rotate( 90, expand=1 )
      image.save(buffered, format="PNG")
      img_str = base64.b64encode(buffered.getvalue())
      if file_serializer.is_valid():
          file_serializer.save()
          return Response({"response": img_str}, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)