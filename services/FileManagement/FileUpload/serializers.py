from rest_framework import serializers
from FileUpload.models import File
class FileSerializer(serializers.ModelSerializer):
    file = serializers.ImageField(max_length=None, use_url=True)
    class Meta:
        model = File
        fields = "__all__"