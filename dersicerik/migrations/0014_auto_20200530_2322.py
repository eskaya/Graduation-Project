# Generated by Django 2.2.12 on 2020-05-30 20:22

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('dersicerik', '0013_auto_20200530_2317'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mesajdersmezun',
            name='dersicerik',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='mesajMezun', to='dersicerik.DersIcerikMezun'),
        ),
    ]
