# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2020-04-28 23:54
from __future__ import unicode_literals

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0005_ogrenci_ilantoplam'),
    ]

    operations = [
        migrations.AddField(
            model_name='mezun',
            name='gorulenIsId',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='mezun',
            name='gorulenStajId',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='mezun',
            name='ilanToplam',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='mezun',
            name='oncekiGirisIs',
            field=models.DateTimeField(default=datetime.datetime(2020, 4, 28, 23, 54, 37, 213671, tzinfo=utc)),
        ),
        migrations.AddField(
            model_name='mezun',
            name='oncekiGirisStaj',
            field=models.DateTimeField(default=datetime.datetime(2020, 4, 28, 23, 54, 37, 213671, tzinfo=utc)),
        ),
    ]
