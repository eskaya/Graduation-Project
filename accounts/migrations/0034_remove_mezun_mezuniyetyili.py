# Generated by Django 2.2.12 on 2020-05-30 20:16

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0033_auto_20200530_2307'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='mezun',
            name='mezuniyetYili',
        ),
    ]
