# Generated by Django 3.0.6 on 2020-06-02 16:34

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('dersicerik', '0016_auto_20200601_0338'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mesajdersmezun',
            name='dersicerik',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='mesajMezun', to='dersicerik.DersIcerikMezun'),
        ),
    ]
