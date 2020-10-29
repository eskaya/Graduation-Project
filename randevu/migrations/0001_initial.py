# Generated by Django 2.2.12 on 2020-05-10 11:35

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='randevuMezun',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('randevuBaslik', models.CharField(max_length=100, verbose_name='Randevu Sebebi')),
                ('randevuIcerik', models.TextField(verbose_name='Randevu İçeriği')),
                ('randevuTarihi', models.DateField()),
                ('randevuTalepTarihi', models.DateTimeField(auto_now_add=True, verbose_name='Randevunun Oluşturulma Tarihi')),
                ('randevuSender', models.CharField(max_length=250)),
                ('mail', models.EmailField(max_length=254)),
                ('ogretimGorevlisi', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='randevuOgrenci',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('randevuBaslik', models.CharField(max_length=100, verbose_name='Randevu Sebebi')),
                ('randevuIcerik', models.TextField(verbose_name='Randevu İçeriği')),
                ('randevuTarihi', models.DateField()),
                ('randevuTalepTarihi', models.DateTimeField(auto_now_add=True, verbose_name='Randevunun Oluşturulma Tarihi')),
                ('randevuSender', models.CharField(max_length=250)),
                ('mail', models.EmailField(max_length=254)),
                ('ogretimGorevlisi', models.CharField(max_length=100)),
            ],
        ),
    ]