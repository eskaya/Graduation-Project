# Generated by Django 2.2.10 on 2020-05-20 18:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('randevu', '0008_auto_20200520_2134'),
    ]

    operations = [
        migrations.AlterField(
            model_name='randevumezun',
            name='ogretimGorevlisi',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='randevuogrenci',
            name='ogretimGorevlisi',
            field=models.CharField(max_length=100),
        ),
    ]
