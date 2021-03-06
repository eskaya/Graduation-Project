# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2020-04-20 03:35
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='IsIlaniEkle',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('isOzeti', models.CharField(max_length=250)),
                ('firma', models.CharField(max_length=250)),
                ('konum', models.CharField(max_length=250)),
                ('ilanTarihi', models.DateField()),
                ('isTanimi', models.TextField()),
                ('nitelikler', models.TextField()),
                ('sektor', models.CharField(max_length=250)),
                ('pozisyon', models.CharField(max_length=250)),
                ('pozisyonSeviyesi', models.CharField(max_length=250)),
                ('calismaSekli', models.CharField(max_length=250)),
                ('adres', models.TextField()),
                ('basvuruSekli', models.CharField(max_length=250)),
                ('sonBasvuruTarihi', models.DateField()),
                ('ilanSahibi', models.CharField(max_length=250)),
                ('ilanSahibiMail', models.EmailField(max_length=254)),
            ],
        ),
        migrations.CreateModel(
            name='stajIlani',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('stajBaslik', models.CharField(max_length=250)),
                ('stajFirma', models.CharField(max_length=250)),
                ('stajKonum', models.CharField(max_length=250)),
                ('stajIlanTarihi', models.DateField()),
                ('stajBilgilendirme', models.TextField()),
                ('stajAdres', models.TextField()),
                ('stajSonBasvuruTarihi', models.DateField()),
                ('stajIlanSahibi', models.CharField(max_length=250)),
                ('stajIlanSahibiMail', models.EmailField(max_length=254)),
            ],
        ),
    ]
