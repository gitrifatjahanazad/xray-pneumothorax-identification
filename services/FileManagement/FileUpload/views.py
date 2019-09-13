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
import cv2
import torch
import pandas as pd
from io import BytesIO
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from FileUpload import predfunc
from FileUpload.predfunc import *
import numpy as np
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
      #print("Pil ", image)
      img = np.array(image)
      img = cv2.resize(img, (512, 512))
      img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
      #print("Image view: ", img)
      image = image.rotate(90, expand=1)
      image.save(buffered, format="PNG")
      img_str = base64.b64encode(buffered.getvalue())

      size = 512
      mean = (0.485, 0.456, 0.406)
      std = (0.229, 0.224, 0.225)
      num_workers = 0
      batch_size = 1
      best_threshold = 0.5
      min_size = 3500
      device = torch.device("cuda:0")
      df = pd.read_csv("FileUpload/dummy.csv")
      td = TestDataset("", df, size, mean, std)
      td.updateImage(img)
      testset = DataLoader(
          td,
          batch_size=batch_size,
          shuffle=False,
          num_workers=num_workers,
          pin_memory=True,
      )
      print(len(testset))
      model = model_trainer.net
      model.eval()
      state = torch.load(
          model_path, map_location=lambda storage, loc: storage)
      model.load_state_dict(state["state_dict"])

      encoded_pixels = []

      for i, batch in enumerate(testset):
        preds = torch.sigmoid(model(batch.to(device)))
            # (batch_size, 1, size, size) -> (batch_size, size, size)
        preds = preds.detach().cpu().numpy()[:, 0, :, :]
        for probability in preds:
            if probability.shape != (1024, 1024):
                probability = cv2.resize(probability, dsize=(
                    1024, 1024), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(
                probability, best_threshold, min_size)
            if num_predict == 0:
                encoded_pixels.append('-1')
            else:
                r = run_length_encode(predict)
                encoded_pixels.append(r)
        df['EncodedPixels'] = encoded_pixels
        df.to_csv('submission.csv', columns=[
                  'ImageId', 'EncodedPixels'], index=False)

        image = cv2.imread(os.path.join(test_data_folder, 'hh.png'))
        print(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        print(image)

        import numpy

        def catch_image(im):
            print("Here I am")
            #im = imagePointer.f
            # image = imagePointer["file"].read()
            # print(image)
            # npimg = numpy.fromstring(image, numpy.uint8)
            # image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
            # image = cv2.imread(image)

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
            # print("Image: ", image)
            #print("Child Node: ", im)
            #filestr = request.files['file'].read()
        #convert string data to numpy array
        #npimg = numpy.fromstring(filestr, numpy.uint8)
        # convert numpy array to image
        #img = cv2.imdecode(npimg, cv2.CV_LOAD_IMAGE_UNCHANGED)
            #print("im: ", im)
      if file_serializer.is_valid():
          file_serializer.save()
          catch_image(img)
          return Response({"response": img_str}, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
