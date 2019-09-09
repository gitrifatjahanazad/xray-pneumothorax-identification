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

@permission_classes((permissions.AllowAny,))
class FileUploadView(APIView):
    parser_class = [FileUploadParser]
    
    def post(self, request, *args, **kwargs):
      file_serializer = FileSerializer(data=request.data)
      print(request.FILES.values)
      print(request.FILES.values())
    #   for f in request.FILES.values():
    #     print(f)
    #     p = ImageFile.Parser()
    #     while 1:
    #         s = f.read(1024)
    #         print("working...")
    #         if not s:
    #             break
    #         p.feed(s)
    #     im = p.close()
    #     im.save("/tmp/" + f.name)
      print(file_serializer.is_valid())
      if file_serializer.is_valid():
          file_serializer.save()
          return Response(file_serializer.data, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)