import matplotlib
from django.shortcuts import redirect, render
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from django.template import Context

from accounts.models import Mezun
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from datetime import date
matplotlib.use('Agg')


def machineLearning(request):
    veriler = Mezun.objects.filter(calismaDurumu='Çalışıyorum')
    sayi=0
    girdi = []
    cikti = []
    for i in veriler:
        if i.iseBaslamaTarihi != None and i.mezuniyetYili != None and i.mezuniyetOrtalamasi != None:
            girdi.append(i.mezuniyetOrtalamasi)
            a = date(i.iseBaslamaTarihi.year, i.iseBaslamaTarihi.month, i.iseBaslamaTarihi.day)
            b = date(i.mezuniyetYili.year, i.mezuniyetYili.month, i.mezuniyetYili.day)
            cikti.append(((a-b).days))
            sayi += 1


    x = np.array(girdi)
    y = np.array(cikti)

    x = x.reshape(sayi,1)
    y = y.reshape(sayi,1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25)


    linearregresyon = lr()
    linearregresyon.fit(X_train, y_train)
    linearregresyonmae = round(mean_absolute_error(y_test, linearregresyon.predict(X_test)), 4)
    linearregresyonmse = round(mean_squared_error(y_test, linearregresyon.predict(X_test)), 4)
    linearregresyonrmse = round(np.sqrt(linearregresyonmse), 4)

    linearregresyonR2 = round(r2_score(y_test, linearregresyon.predict(X_test)), 4)
    rmse = mean_squared_error(y_test, linearregresyon.predict(X_test))
    r2 = r2_score(y_test, linearregresyon.predict(X_test))

    linearregresyon.predict(X_train)
    m = linearregresyon.coef_
    b = linearregresyon.intercept_
    a = np.arange(50, 100)
    plt.scatter(X_train, y_train, c='blue')
    plt.plot(X_train, linearregresyon.predict(X_train), c='green')
    plt.title("Mezuniyet Ortalaması-İşe Başlama Süresi")
    plt.ylabel("İşe Başlama Süresi(Gün)")
    plt.xlabel("Mezuniyet Ortalaması")
    plt.savefig('static/grafikler/grafik2.png')
    plt.close('all')

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25)

    poly_reg = PolynomialFeatures(degree=3)
    X_poly = poly_reg.fit_transform(X_train)
    linearregresyon2 = lr()
    linearregresyon2.fit(X_poly, y_train)

    plt.scatter(X_train, y_train, c='blue')
    X_train.sort(axis=0)
    plt.plot(X_train, linearregresyon2.predict(poly_reg.fit_transform(X_train)), c='green')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25)
    plt.title("Mezuniyet Ortalaması-İşe Başlama Süresi")
    plt.ylabel("İşe Başlama Süresi(Gün)")
    plt.xlabel("Mezuniyet Ortalaması")
    plt.savefig('static/grafikler/grafik.png')
    plt.close('all')

    y_test_predict = linearregresyon2.predict(poly_reg.fit_transform(X_test))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
    r2_test = r2_score(y_test, y_test_predict)

    polinomikresresyonR2 = round(linearregresyon2.score(poly_reg.fit_transform(X_test), y_test), 4)
    polinomikresresyonmae = round(mean_absolute_error(y_test, linearregresyon2.predict(poly_reg.fit_transform(X_test))), 4)
    polinomikresresyonmse = round(mean_squared_error(y_test, linearregresyon2.predict(poly_reg.fit_transform(X_test))), 4)
    polinomikresresyonrmse = round(np.sqrt(polinomikresresyonmse), 4)



    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=25)

    regressor = DecisionTreeRegressor(max_depth=None, random_state=25)
    regressor.fit(X_train, y_train)
    X_grid = np.arange(min(X_train), max(X_train), 0.1)
    X_grid = X_grid.reshape((len(X_grid),1))
    plt.scatter(X_train, y_train, c='blue')
    plt.plot(X_grid, regressor.predict(X_grid), c='red')
    plt.title("Mezuniyet Ortalaması-İşe Başlama Süresi")
    plt.ylabel("İşe Başlama Süresi(Gün)")
    plt.xlabel("Mezuniyet Ortalaması")
    plt.savefig('static/grafikler/grafik3.png')
    plt.close('all')
    r2_skor = metrics.r2_score(y_test, regressor.predict(X_test))

    kararagacıR2 = round(r2_score(y_test, regressor.predict(X_test)), 4)
    kararagacımae = round(mean_absolute_error(y_test, regressor.predict(X_test)), 4)
    kararagacımse = round(mean_squared_error(y_test, regressor.predict(X_test)),4 )
    kararagacırmse = round(np.sqrt(kararagacımse), 4)


    randomforest = RandomForestRegressor(n_estimators=10, random_state=25)
    randomforest.fit(X_train, y_train.ravel())
    X_grid = np.arange(min(X_train), max(X_train), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_train, y_train, c='blue')
    plt.plot(X_grid, randomforest.predict(X_grid), c='red')
    plt.title("Mezuniyet Ortalaması-İşe Başlama Süresi")
    plt.ylabel("İşe Başlama Süresi(Gün)")
    plt.xlabel("Mezuniyet Ortalaması")
    plt.savefig('static/grafikler/grafik4.png')
    plt.close('all')
    r2_skor = metrics.r2_score(y_test, randomforest.predict(X_test))
    randomforestR2 = round(r2_score(y_test, randomforest.predict(X_test)), 4)
    randomforestmae = round(mean_absolute_error(y_test, randomforest.predict(X_test)), 4)
    randomforestmse = round(mean_squared_error(y_test, randomforest.predict(X_test)), 4)
    randomforestrmse = round(np.sqrt(randomforestmse), 4)

    sonuc = 0; sonuc2=0; sonuc3=0; sonuc4=0; ortalama=0
    if request.method == 'POST':
        if request.POST.get('submit') == 'grafik':
            deger = float(request.POST.get('grafik'))
            if deger != None:
                ortalama = deger
                sonuc = int(linearregresyon2.predict(poly_reg.fit_transform([[deger]]))[0][0])
                sonuc2 = int(linearregresyon.predict([[deger]])[0][0])
                sonuc3 = int(regressor.predict([[deger]]))
                sonuc4 = int(randomforest.predict([[deger]]))




    veriler2 = Mezun.objects.filter(calismaDurumu='Çalışıyorum')
    sayi2 = 0
    girdi2 = []
    cikti2 = []
    for i in veriler2:
        if i.iseBaslamaTarihi != None and i.mezuniyetYili != None:
            if i.mezuniyetYili.year > 2000 and i.mezuniyetYili.year <2012:

                girdi2.append(i.mezuniyetYili.year)
                a = date(i.iseBaslamaTarihi.year, i.iseBaslamaTarihi.month, i.iseBaslamaTarihi.day)
                b = date(i.mezuniyetYili.year, i.mezuniyetYili.month, i.mezuniyetYili.day)
                cikti2.append(((a - b).days))
                sayi2 += 1

    x2 = np.array(girdi2)
    y2 = np.array(cikti2)

    x2 = x2.reshape(sayi2, 1)
    y2 = y2.reshape(sayi2, 1)

    X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size=0.3, random_state=25)

    linearregresyon3 = lr()
    linearregresyon3.fit(X_train, y_train)
    linearregresyonmae2 = round(mean_absolute_error(y_test, linearregresyon3.predict(X_test)), 4)
    linearregresyonmse2 = round(mean_squared_error(y_test, linearregresyon3.predict(X_test)), 4)
    linearregresyonrmse2 = round(np.sqrt(linearregresyonmse2), 4)

    linearregresyonR22 = round(r2_score(y_test, linearregresyon3.predict(X_test)), 4)

    linearregresyon3.predict(X_train)
    m2 = linearregresyon3.coef_
    b2 = linearregresyon3.intercept_
    a2 = np.arange(min(X_train), max(X_train))
    plt.scatter(X_train, y_train)
    plt.plot(X_train, linearregresyon3.predict(X_train))
    plt.title("Mezuniyet Yılı-İşe Başlama Süresi")
    plt.ylabel("İşe Başlama Süresi(Gün)")
    plt.xlabel("Mezuniyet Yılı")
    plt.savefig('static/grafikler/grafik5.png')
    plt.close('all')

    X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size=0.3, random_state=25)

    poly_reg2 = PolynomialFeatures(degree=3)
    X_poly2 = poly_reg2.fit_transform(X_train)
    linearregresyon4 = lr()
    linearregresyon4.fit(X_poly2, y_train)

    plt.scatter(X_train, y_train, c='blue')
    X_train.sort(axis=0)
    plt.plot(X_train, linearregresyon4.predict(poly_reg2.fit_transform(X_train)), c='green')
    X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size=0.3, random_state=25)
    plt.title("Mezuniyet Yılı-İşe Başlama Süresi")
    plt.ylabel("İşe Başlama Süresi(Gün)")
    plt.xlabel("Mezuniyet Yılı")
    plt.savefig('static/grafikler/grafik6.png')
    plt.close('all')

    y_test_predict2 = linearregresyon4.predict(poly_reg2.fit_transform(X_test))

    polinomikresresyonR22 = round(linearregresyon4.score(poly_reg2.fit_transform(X_test), y_test), 4)
    polinomikresresyonmae2 = round(mean_absolute_error(y_test, linearregresyon4.predict(poly_reg2.fit_transform(X_test))),
                                  4)
    polinomikresresyonmse2 = round(mean_squared_error(y_test, linearregresyon4.predict(poly_reg2.fit_transform(X_test))),
                                  4)
    polinomikresresyonrmse2 = round(np.sqrt(polinomikresresyonmse2), 4)

    X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size=0.3, random_state=25)

    regressor2 = DecisionTreeRegressor(max_depth=None, random_state=25)
    regressor2.fit(X_train, y_train)
    X_grid2 = np.arange(min(X_train), max(X_train), 0.1)
    X_grid2 = X_grid2.reshape((len(X_grid2), 1))
    plt.scatter(X_train, y_train, c='blue')
    plt.plot(X_grid2, regressor2.predict(X_grid2), c='red')
    plt.title("Mezuniyet Yılı-İşe Başlama Süresi")
    plt.ylabel("İşe Başlama Süresi(Gün)")
    plt.xlabel("Mezuniyet Yılı")
    plt.savefig('static/grafikler/grafik7.png')
    plt.close('all')

    kararagacıR22 = round(r2_score(y_test, regressor2.predict(X_test)), 4)
    kararagacımae2 = round(mean_absolute_error(y_test, regressor2.predict(X_test)), 4)
    kararagacımse2 = round(mean_squared_error(y_test, regressor2.predict(X_test)), 4)
    kararagacırmse2 = round(np.sqrt(kararagacımse2), 4)

    randomforest2 = RandomForestRegressor(n_estimators=10, random_state=25)
    randomforest2.fit(X_train, y_train.ravel())
    X_grid2 = np.arange(min(X_train), max(X_train), 0.1)
    X_grid2 = X_grid2.reshape((len(X_grid2), 1))
    plt.scatter(X_train, y_train, c='blue')
    plt.plot(X_grid2, randomforest2.predict(X_grid2), c='red')
    plt.title("Mezuniyet Yılı-İşe Başlama Süresi")
    plt.ylabel("İşe Başlama Süresi(Gün)")
    plt.xlabel("Mezuniyet Yılı")
    plt.savefig('static/grafikler/grafik8.png')
    plt.close('all')

    randomforestR22 = round(r2_score(y_test, randomforest2.predict(X_test)), 4)
    randomforestmae2 = round(mean_absolute_error(y_test, randomforest2.predict(X_test)), 4)
    randomforestmse2 = round(mean_squared_error(y_test, randomforest2.predict(X_test)), 4)
    randomforestrmse2 = round(np.sqrt(randomforestmse2), 4)

    sonuc21 = 0;
    sonuc22 = 0;
    sonuc23 = 0;
    sonuc24 = 0;
    yil = 0
    if request.method == 'POST':
        if request.POST.get('submit') == 'grafik2':
            deger2 = int(request.POST.get('grafik2'))
            print(deger2)
            print([[deger2]])
            if deger2 != None:
                yil = deger2
                sonuc21 = int(linearregresyon4.predict(poly_reg2.fit_transform([[deger2]]))[0][0])
                sonuc22 = int(linearregresyon3.predict([[deger2]])[0][0])
                sonuc23 = int(regressor2.predict([[deger2]]))
                sonuc24 = int(randomforest2.predict([[deger2]]))

    veriler3 = Mezun.objects.filter(calismaDurumu='Çalışıyorum')
    sayi3 = 0
    girdi3 = []
    cikti3 = []
    for i in veriler3:
        if i.maas != None and i.mezuniyetYili != None:
            if i.mezuniyetYili.year > 2000 and i.mezuniyetYili.year < 2012:
                girdi3.append(i.mezuniyetYili.year)
                cikti3.append(int(i.maas))
                sayi3 += 1

    x3 = np.array(girdi3)
    y3 = np.array(cikti3)

    x3 = x3.reshape(sayi3, 1)
    y3 = y3.reshape(sayi3, 1)

    X_train, X_test, y_train, y_test = train_test_split(x3, y3, test_size=0.3, random_state=25)

    linearregresyon5 = lr()
    linearregresyon5.fit(X_train, y_train)
    linearregresyonmae3 = round(mean_absolute_error(y_test, linearregresyon5.predict(X_test)), 4)
    linearregresyonmse3 = round(mean_squared_error(y_test, linearregresyon5.predict(X_test)), 4)
    linearregresyonrmse3 = round(np.sqrt(linearregresyonmse3), 4)

    linearregresyonR23 = round(r2_score(y_test, linearregresyon5.predict(X_test)), 4)

    linearregresyon5.predict(X_train)
    m2 = linearregresyon5.coef_
    b2 = linearregresyon5.intercept_
    a2 = np.arange(min(X_train), max(X_train))
    plt.scatter(X_train, y_train)
    plt.plot(X_train, linearregresyon5.predict(X_train))
    plt.title("Mezuniyet Yılı-Maaş")
    plt.ylabel("Maaş (₺)")
    plt.xlabel("Mezuniyet Yılı")
    plt.savefig('static/grafikler/grafik9.png')
    plt.close('all')

    X_train, X_test, y_train, y_test = train_test_split(x3, y3, test_size=0.3, random_state=25)

    poly_reg3 = PolynomialFeatures(degree=3)
    X_poly3 = poly_reg3.fit_transform(X_train)
    linearregresyon6 = lr()
    linearregresyon6.fit(X_poly3, y_train)

    plt.scatter(X_train, y_train, c='blue')
    X_train.sort(axis=0)
    plt.plot(X_train, linearregresyon6.predict(poly_reg3.fit_transform(X_train)), c='green')
    X_train, X_test, y_train, y_test = train_test_split(x3, y3, test_size=0.3, random_state=25)
    plt.title("Mezuniyet Yılı-Maaş")
    plt.ylabel("Maaş (₺)")
    plt.xlabel("Mezuniyet Yılı")
    plt.savefig('static/grafikler/grafik10.png')
    plt.close('all')

    y_test_predict2 = linearregresyon6.predict(poly_reg3.fit_transform(X_test))

    polinomikresresyonR23 = round(linearregresyon6.score(poly_reg3.fit_transform(X_test), y_test), 4)
    polinomikresresyonmae3 = round(
        mean_absolute_error(y_test, linearregresyon6.predict(poly_reg3.fit_transform(X_test))),
        4)
    polinomikresresyonmse3 = round(
        mean_squared_error(y_test, linearregresyon6.predict(poly_reg3.fit_transform(X_test))),
        4)
    polinomikresresyonrmse3 = round(np.sqrt(polinomikresresyonmse3), 4)

    X_train, X_test, y_train, y_test = train_test_split(x3, y3, test_size=0.3, random_state=25)

    regressor3 = DecisionTreeRegressor(max_depth=None, random_state=25)
    regressor3.fit(X_train, y_train)
    X_grid3 = np.arange(min(X_train), max(X_train), 0.1)
    X_grid3 = X_grid3.reshape((len(X_grid3), 1))
    plt.scatter(X_train, y_train, c='blue')
    plt.plot(X_grid3, regressor3.predict(X_grid3), c='red')
    plt.title("Mezuniyet Yılı-Maaş")
    plt.ylabel("Maaş (₺)")
    plt.xlabel("Mezuniyet Yılı")
    plt.savefig('static/grafikler/grafik11.png')
    plt.close('all')

    kararagacıR23 = round(r2_score(y_test, regressor3.predict(X_test)), 4)
    kararagacımae3 = round(mean_absolute_error(y_test, regressor3.predict(X_test)), 4)
    kararagacımse3 = round(mean_squared_error(y_test, regressor3.predict(X_test)), 4)
    kararagacırmse3 = round(np.sqrt(kararagacımse3), 4)

    randomforest3 = RandomForestRegressor(n_estimators=10, random_state=25)
    randomforest3.fit(X_train, y_train.ravel())
    X_grid3 = np.arange(min(X_train), max(X_train), 0.1)
    X_grid3 = X_grid3.reshape((len(X_grid3), 1))
    plt.scatter(X_train, y_train, c='blue')
    plt.plot(X_grid3, randomforest3.predict(X_grid3), c='red')
    plt.title("Mezuniyet Yılı-Maaş")
    plt.ylabel("Maaş (₺)")
    plt.xlabel("Mezuniyet Yılı")
    plt.savefig('static/grafikler/grafik12.png')
    plt.close('all')

    randomforestR23 = round(r2_score(y_test, randomforest3.predict(X_test)), 4)
    randomforestmae3 = round(mean_absolute_error(y_test, randomforest3.predict(X_test)), 4)
    randomforestmse3 = round(mean_squared_error(y_test, randomforest3.predict(X_test)), 4)
    randomforestrmse3 = round(np.sqrt(randomforestmse3), 4)

    sonuc31 = 0;
    sonuc32 = 0;
    sonuc33 = 0;
    sonuc34 = 0;
    yil2 = 0
    if request.method == 'POST':
        if request.POST.get('submit') == 'grafik3':
            deger3 = int(request.POST.get('grafik3'))

            if deger3 != None:
                yil2 = deger3
                sonuc31 = int(linearregresyon6.predict(poly_reg3.fit_transform([[deger3]]))[0][0])
                sonuc32 = int(linearregresyon5.predict([[deger3]])[0][0])
                sonuc33 = int(regressor3.predict([[deger3]]))
                sonuc34 = int(randomforest3.predict([[deger3]]))


    return render(request, 'machine/machine.html', {'sonuc':sonuc, 'sonuc2':sonuc2, 'sonuc3':sonuc3, 'sonuc4':sonuc4, 'ortalama':ortalama,
                                                    'linearregresyonR2':linearregresyonR2, 'polinomikresresyonR2':polinomikresresyonR2,
                                                    'kararagacıR2':kararagacıR2, 'randomforestR2':randomforestR2,
                                                    'randomforestmae': randomforestmae, 'randomforestmse':randomforestmse, 'randomforestrmse':randomforestrmse,
                                                    'kararagacımae':kararagacımae, 'kararagacımse':kararagacımse, 'kararagacırmse':kararagacırmse,
                                                    'polinomikresresyonmae':polinomikresresyonmae, 'polinomikresresyonmse':polinomikresresyonmse, 'polinomikresresyonrmse':polinomikresresyonrmse,
                                                    'linearregresyonmae':linearregresyonmae, 'linearregresyonmse':linearregresyonmse, 'linearregresyonrmse':linearregresyonrmse,
                                                    'linearregresyonR22': linearregresyonR22,
                                                    'polinomikresresyonR22': polinomikresresyonR22,
                                                    'kararagacıR22': kararagacıR22, 'randomforestR22': randomforestR22,
                                                    'randomforestmae2': randomforestmae2,
                                                    'randomforestmse2': randomforestmse2,
                                                    'randomforestrmse2': randomforestrmse2,
                                                    'kararagacımae2': kararagacımae2, 'kararagacımse2': kararagacımse2,
                                                    'kararagacırmse2': kararagacırmse2,
                                                    'polinomikresresyonmae2': polinomikresresyonmae2,
                                                    'polinomikresresyonmse2': polinomikresresyonmse2,
                                                    'polinomikresresyonrmse2': polinomikresresyonrmse2,
                                                    'linearregresyonmae2': linearregresyonmae2,
                                                    'linearregresyonmse2': linearregresyonmse2,
                                                    'linearregresyonrmse2': linearregresyonrmse2,
                                                    'sonuc21': sonuc21, 'sonuc22': sonuc22, 'sonuc23': sonuc23,
                                                    'sonuc24': sonuc24, 'yil': yil,
                                                    'linearregresyonR23': linearregresyonR23,
                                                    'polinomikresresyonR23': polinomikresresyonR23,
                                                    'kararagacıR23': kararagacıR23, 'randomforestR23': randomforestR23,
                                                    'randomforestmae3': randomforestmae3,
                                                    'randomforestmse3': randomforestmse3,
                                                    'randomforestrmse3': randomforestrmse3,
                                                    'kararagacımae3': kararagacımae3, 'kararagacımse3': kararagacımse3,
                                                    'kararagacırmse3': kararagacırmse3,
                                                    'polinomikresresyonmae3': polinomikresresyonmae3,
                                                    'polinomikresresyonmse3': polinomikresresyonmse3,
                                                    'polinomikresresyonrmse3': polinomikresresyonrmse3,
                                                    'linearregresyonmae3': linearregresyonmae3,
                                                    'linearregresyonmse3': linearregresyonmse3,
                                                    'linearregresyonrmse3': linearregresyonrmse3,
                                                    'sonuc31': sonuc31, 'sonuc32': sonuc32, 'sonuc33': sonuc33,
                                                    'sonuc34': sonuc34, 'yil2': yil2
                                                    })

