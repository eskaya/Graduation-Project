# Generated by Django 2.2.12 on 2020-05-15 22:41

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0024_remove_mezun_ulke'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='ogretimgorevlisi',
            name='gorulenDersIcerik',
        ),
        migrations.RemoveField(
            model_name='ogretimgorevlisi',
            name='oncekiGirisDersIcerik',
        ),
    ]
