# Generated by Django 2.2.12 on 2020-05-16 18:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('randevu', '0004_auto_20200516_2100'),
    ]

    operations = [
        migrations.AddField(
            model_name='mesajmezun',
            name='randevu',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='mesajMezun', to='randevu.randevuMezun'),
        ),
        migrations.AddField(
            model_name='mesajogrenci',
            name='randevu',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='mesajOgrenci', to='randevu.randevuOgrenci'),
        ),
    ]
