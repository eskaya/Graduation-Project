# Generated by Django 2.2.12 on 2020-05-16 18:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('randevu', '0003_auto_20200516_2058'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='mesajmezun',
            name='randevu',
        ),
        migrations.RemoveField(
            model_name='mesajogrenci',
            name='randevu',
        ),
    ]