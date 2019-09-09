from django.db import models

class File(models.Model):
    file = models.ImageField(upload_to="media/",default="media/None/no-doc.png")
    def __str__(self):
        return self.file.name
