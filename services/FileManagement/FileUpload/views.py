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
      img = np.array(image)

      img = cv2.resize(img, (1024, 1024))
      img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
      
      img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
      print("Image view: ", img)
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
        print("Encoded Pixels: ", df.head(5))

        
        # MASKING
        def rle2mask(rle, width, height):
            mask= np.zeros(width* height)
            #mask = image
            array = np.asarray([int(x) for x in rle.split()])
            starts = array[0::2]
            lengths = array[1::2]

            current_position = 0
            for index, start in enumerate(starts):
                current_position += start
                mask[current_position:current_position+lengths[index]] = 255
                current_position += lengths[index]

            return mask.reshape(width, height)

        masks = df['EncodedPixels']
        print(masks.head())

        img1 = np.zeros((1024,1024))
        if(type(masks)!=str or (type(masks) == str and masks!='-1')):
            if(type(masks) == str): masks = [masks]
            else: 
                masks = masks.tolist()
            for mask in masks:
                img1 += rle2mask(mask, 1024, 1024).T
            img1 = cv2.resize(img1, (1024, 1024))
            out = cv2.imencode('.png',img1)[1]
            img_str = base64.b64encode(out)
          

      if file_serializer.is_valid():
          file_serializer.save()
          #catch_image(img)
          return Response({"response": img_str}, status=status.HTTP_201_CREATED)
      else:
          return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
