from PyQt5 import QtWidgets, uic,QtGui, QtCore
from PyQt5.QtGui import QStandardItem, QDoubleValidator
from PyQt5.QtCore import QAbstractTableModel, Qt
import sys
import os
import pandas as pd
from statistics import linear_regression
from scipy.optimize import curve_fit
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from shapely.geometry import LineString
import qdarkstyle
import numpy as np

qtcreator_file = "mat_cal.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)
class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        model = QtGui.QStandardItemModel()
        self.tabd.setModel(model)
        self.la.clicked.connect(self.OpenFile)
        self.clr.clicked.connect(self.Clear)
        self.ru.clicked.connect(self.pri)
        self.expo.clicked.connect(self.expot)
        self.expo.setEnabled(False)
        self.tabWidget_3.setTabVisible(0, False)
        self.tabWidget_3.setTabVisible(1, False)
        self.erc.clicked.connect(self.dyc)
        self.ru_2.clicked.connect(self.evalx)
        self.refst.clicked.connect(self.refop)
        self.refst_2.clicked.connect(self.refop_1)
        self.refst_3.clicked.connect(self.refop_2)
        self.refst_4.clicked.connect(self.refop_3)
        self.refst_5.clicked.connect(self.refop_4)
        self.refst_6.clicked.connect(self.refop_5)
        self.mox.clicked.connect(self.rep)
        self.mox_2.setVisible(False)
        self.mox_2.clicked.connect(self.zenza)
        self.ru_2.setVisible(False)
        self.ru.setVisible(False)
        self.tabWidget.setTabVisible(0, True)
        self.tabWidget.setTabVisible(1, False)
        self.tabWidget.setTabVisible(2, False)
        self.tabWidget.setTabVisible(3, False)
        self.tabWidget.setTabVisible(4, False)
        self.tabWidget.setTabVisible(5, False)
        self.tabWidget.setTabVisible(6, False)
        self.tabWidget.setTabVisible(7, False)
        self.tabWidget_2.setTabEnabled(2,False)
        self.pg.setValue(0)
        self.ref1.setValidator(QDoubleValidator())
        self.ref2.setValidator(QDoubleValidator())
        self.ref3.setValidator(QDoubleValidator())
        self.ref4.setValidator(QDoubleValidator())
        self.ref5.setValidator(QDoubleValidator())
        self.ref6.setValidator(QDoubleValidator())
        self.den.setValidator(QDoubleValidator())
        self.nux.setValidator(QDoubleValidator())

    def dyc(self):
        self.tabWidget_4.setTabVisible(0,False)
        self.tabWidget_4.setTabVisible(1, False)
        self.mox.setVisible(False)
        self.ru_2.setVisible(True)
        self.ru.setVisible(False)
        self.mox_2.setVisible(True)
        self.tabWidget_3.setTabVisible(0, False)
        self.tabWidget_3.setTabVisible(1, True)
        self.la.setEnabled(False)

        self.tabWidget.setTabVisible(0, False)
        self.tabWidget.setTabVisible(1, False)
        self.tabWidget.setTabVisible(2, False)
        self.tabWidget.setTabVisible(3, False)
        self.tabWidget.setTabVisible(4, False)
        self.tabWidget.setTabVisible(5, False)
        self.tabWidget.setTabVisible(6, True)
        global all_data1
        all_data1=pd.DataFrame(np.zeros((1, 2)),columns=['Strain','Stress'])
        global all_data2
        all_data2 = pd.DataFrame(np.zeros((1, 2)),columns=['Strain','Stress'])
        global all_data3
        all_data3 = pd.DataFrame(np.zeros((1, 2)),columns=['Strain','Stress'])
        global all_data4
        all_data4 = pd.DataFrame(np.zeros((1, 2)),columns=['Strain','Stress'])
        global all_data5
        all_data5 = pd.DataFrame(np.zeros((1, 2)),columns=['Strain','Stress'])
        global all_data6
        all_data6 = pd.DataFrame(np.zeros((1, 2)),columns=['Strain','Stress'])

    def refop(self):
        try:
            global all_data1
            global c1
            path = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('Home'), 'CSV(*.csv)')
            all_data1 = pd.read_csv(path[0])
            if path[0] == '':
                return 0
            name = (path[0])
            self.lineEdit_2.setText(name)
            c2 = self.ref1.text()

        except FileNotFoundError:
            all_data1 = pd.DataFrame(np.zeros((1, 2)), columns=['Strain', 'Stress'])
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("File Not Chosen")

            def msgButtonClick(i):
                msg.close()

            msg.setStandardButtons(QMessageBox.Cancel)
            msg.show()
            msg.buttonClicked.connect(msgButtonClick)

    def refop_1(self):
        try:
            global all_data2
            global c2
            path = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('Home'), 'CSV(*.csv)')
            all_data2 = pd.read_csv(path[0])
            if path[0] == '':
                return 0
            name = (path[0])
            self.lineEdit_4.setText(name)
            c2 = self.ref2.text()
        except FileNotFoundError:
            all_data2=pd.DataFrame(np.zeros((1, 2)),columns=['Strain','Stress'])
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("File Not Chosen")

            def msgButtonClick(i):
                msg.close()

            msg.setStandardButtons(QMessageBox.Cancel)
            msg.show()
            msg.buttonClicked.connect(msgButtonClick)

    def refop_2(self):
        try:
            global c3
            global all_data3
            path = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('Home'), 'CSV(*.csv)')
            all_data3 = pd.read_csv(path[0])
            if path[0] == '':
                return 0
            name = (path[0])
            self.lineEdit_6.setText(name)
            c3 = self.ref3.text()
        except FileNotFoundError:
            all_data3 = pd.DataFrame(np.zeros((1, 2)), columns=['Strain', 'Stress'])
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("File Not Chosen")

            def msgButtonClick(i):
                msg.close()

            msg.setStandardButtons(QMessageBox.Cancel)
            msg.show()
            msg.buttonClicked.connect(msgButtonClick)

    def refop_3(self):
        try:
            global c4
            global all_data4
            # self.tabWidget_3.setTabVisible(0, True)
            # self.tabWidget_3.setTabVisible(1, False)
            path = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('Home'), 'CSV(*.csv)')
            all_data4 = pd.read_csv(path[0])
            if path[0] == '':
                return 0
            name = (path[0])
            self.lineEdit_8.setText(name)
            c4 = self.ref4.text()
        except FileNotFoundError:
            all_data4 = pd.DataFrame(np.zeros((1, 2)), columns=['Strain', 'Stress'])
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("File Not Chosen")

            def msgButtonClick(i):
                msg.close()

            msg.setStandardButtons(QMessageBox.Cancel)
            msg.show()
            msg.buttonClicked.connect(msgButtonClick)

    def refop_4(self):
        try:
            global c5
            global all_data5
            path = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('Home'), 'CSV(*.csv)')
            all_data5 = pd.read_csv(path[0])
            if path[0] == '':
                return 0
            name = (path[0])
            self.lineEdit_10.setText(name)
            c5 = self.ref5.text()
        except FileNotFoundError:
            all_data5 = pd.DataFrame(np.zeros((1, 2)), columns=['Strain', 'Stress'])
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("File Not Chosen")

            def msgButtonClick(i):
                msg.close()

            msg.setStandardButtons(QMessageBox.Cancel)
            msg.show()
            msg.buttonClicked.connect(msgButtonClick)

    def refop_5(self):
        try:
            global c6
            global  all_data6
            path = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('Home'), 'CSV(*.csv)')
            all_data6 = pd.read_csv(path[0])
            if path[0] == '':
                return 0
            name = (path[0])
            self.lineEdit_12.setText(name)
            c6 = self.ref6.text()
        except FileNotFoundError:
            all_data6 = pd.DataFrame(np.zeros((1, 2)), columns=['Strain', 'Stress'])
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("File Not Chosen")

            def msgButtonClick(i):
                msg.close()

            msg.setStandardButtons(QMessageBox.Cancel)
            msg.show()
            msg.buttonClicked.connect(msgButtonClick)

    def OpenFile(self):
     try:
        self.erc.setEnabled(False)
        global all_data
        self.tabWidget_3.setTabVisible(0, True)
        self.tabWidget_3.setTabVisible(1, False)
        path=QFileDialog.getOpenFileName(self,'Open CSV',os.getenv('Home'),'CSV(*.csv)')
        all_data=pd.read_csv(path[0])
        if path[0] == '':
            return 0
        self.pg.setValue(10)
        model=pandasModel(all_data)
        self.tabd.setModel(model)
        self.ru.setEnabled(True)
        self.ru.setVisible(True)
        self.ru_2.setVisible(False)
        self.tabWidget.setTabVisible(0, False)
        self.tabWidget.setTabVisible(3, False)

     except FileNotFoundError:
        all_data=pd.DataFrame()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Warning")
        msg.setText("File Not Chosen")

        def msgButtonClick(i):
          msg.close()
        msg.setStandardButtons(QMessageBox.Cancel)
        msg.show()
        msg.buttonClicked.connect(msgButtonClick)

    def Clear(self):
        self.pg.setValue(100)
        self.sia.clear()
        self.mplwidget.canvas.axes.clear()
        self.mplwidget.canvas.axes2.clear()
        self.mplwidget.canvas.axes3.clear()
        self.mplwidget.canvas.axes4.clear()
        self.mplwidget.canvas.draw()
        self.mplwidget_3.canvas.axes.clear()
        self.mplwidget_3.canvas.draw()
        self.mplwidget_4.canvas.axes.clear()
        self.mplwidget_4.canvas.draw()
        self.mplwidget_5.canvas.axes.clear()
        self.mplwidget_5.canvas.draw()
        self.mplwidget_6.canvas.axes.clear()
        self.mplwidget_6.canvas.draw()
        self.mplwidget_7.canvas.axes.clear()
        self.mplwidget_7.canvas.draw()
        self.tabWidget_3.setTabVisible(0, False)
        self.tabWidget_3.setTabVisible(1, False)
        self.tabWidget.setTabVisible(0, True)
        self.tabWidget.setTabVisible(1, False)
        self.tabWidget.setTabVisible(2, False)
        self.tabWidget.setTabVisible(3, False)
        self.tabWidget.setTabVisible(4, False)
        self.tabWidget.setTabVisible(5, False)
        self.tabWidget.setTabVisible(6, False)
        self.tabWidget.setTabVisible(7, False)
        self.tabWidget_2.setTabEnabled(2, False)
        self.erc.setEnabled(True)
        self.la.setEnabled(True)
        self.ru.setEnabled(False)
        self.expo.setEnabled(False)
        self.ru_2.setVisible(False)
        self.ru.setVisible(False)
        model = QtGui.QStandardItemModel()
        self.tabd.setModel(model)
        indices = self.tabd.selectionModel().selectedRows()
        for index in sorted(indices):
            model.removeRow(index.row())
        self.dym.setModel(model)
        indices2 = self.dym.selectionModel().selectedRows()
        for index in sorted(indices2):
            model.removeRow(index.row())
        self.pg.setValue(0)

    def pri(self):
        self.tabWidget_3.setTabVisible(0, True)
        self.tabWidget_3.setTabVisible(1, False)
        self.tabWidget.setTabVisible(0, False)
        self.tabWidget.setTabVisible(1, True)
        self.tabWidget.setTabVisible(2, True)
        self.tabWidget.setTabVisible(3, True)
        self.tabWidget.setTabVisible(4, True)
        self.tabWidget.setTabVisible(5, True)
        self.tabWidget.setTabVisible(6, False)
        def exi(x, k, a1, n):
            return k * pow((a1 + x), n)

        def linear(X, a):
            return a * X

        def rsq(Ypre, Y):
            y1 = sum(pow(Ypre, 2))
            y2 = len(Y) * np.var(Y)
            return 1 - (y1 / y2)

        def chi(X, Y, p1):
            residuals = Y - linear(X, *p1)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((Y3 - np.mean(Y3)) ** 2)
            score = 1 - (ss_res / ss_tot)
            return score
        ds= all_data
        X = ds['Strain'].to_numpy()
        Y = ds['Stress'].to_numpy()

        halfdf = len(ds) // 2
        first_half = ds.iloc[:halfdf]
        secondhalf = len(first_half) // 2
        second_half = ds.iloc[:secondhalf]
        thirdhalf = len(second_half) // 2
        third_half = ds.iloc[:thirdhalf]

        X1 = first_half['Strain'].to_numpy()
        Y1 = first_half['Stress'].to_numpy()
        X2 = second_half['Strain'].to_numpy()
        Y2 = second_half['Stress'].to_numpy()
        X3 = third_half['Strain'].to_numpy()
        Y3 = third_half['Stress'].to_numpy()

        maxfe = 100000

        slope, intercept = curve_fit(linear, X, Y, maxfev=maxfe)
        slope1, intercept1 = curve_fit(linear, X1, Y1, maxfev=maxfe)
        slope2, intercept2 = curve_fit(linear, X2, Y2, maxfev=maxfe)
        slope3, intercept3 = curve_fit(linear, X3, Y3, maxfev=maxfe)
        self.pg.setValue(20)

        a1 = chi(X, Y, slope)
        a2 = chi(X1, Y1, slope1)
        a3 = chi(X2, Y2, slope3)
        a4 = chi(X3, Y3, slope3)
        self.pg.setValue(30)

        self.mplwidget.canvas.axes.clear()
        self.mplwidget.canvas.axes.set_xlabel("Strain")
        self.mplwidget.canvas.axes.set_ylabel("Stress")
        self.mplwidget.canvas.axes.scatter(X,Y,color='orange')
        self.mplwidget.canvas.axes.scatter(X1, Y1, color='red')
        self.mplwidget.canvas.axes.scatter(X2, Y2, color='blue')
        self.mplwidget.canvas.axes.scatter(X3, Y3, color='green')
        #self.mplwidget.canvas.axes.plot(X, slope * X, color='black')
        #self.mplwidget.canvas.axes.legend(loc='upper left')
        self.mplwidget.canvas.axes.set_title('Master Curve Split')
        self.mplwidget.canvas.draw()



        self.mplwidget.canvas.axes2.clear()
        self.mplwidget.canvas.axes2.set_xlabel("Strain")
        self.mplwidget.canvas.axes2.set_ylabel("Stress")
        self.mplwidget.canvas.axes2.scatter(X1, Y1, color='red')
        self.mplwidget.canvas.axes2.plot(X1, slope1 * X1, color='black', label='E=%0.3f (R2 = %0.5f)' % (slope1, a2))
        self.mplwidget.canvas.axes2.legend(loc='upper left')
        self.mplwidget.canvas.axes2.set_title('Split-1')
        self.mplwidget.canvas.draw()


        self.mplwidget.canvas.axes3.clear()
        self.mplwidget.canvas.axes3.set_xlabel("Strain")
        self.mplwidget.canvas.axes3.set_ylabel("Stress")
        self.mplwidget.canvas.axes3.scatter(X2, Y2, color='blue')
        self.mplwidget.canvas.axes3.plot(X2, slope2 * X2, color='black', label='E=%0.3f(R2 = %0.5f)' % (slope2, a3))
        self.mplwidget.canvas.axes3.legend(loc='upper left')
        self.mplwidget.canvas.axes3.set_title('Split-2')
        self.mplwidget.canvas.draw()


        self.mplwidget.canvas.axes4.clear()
        self.mplwidget.canvas.axes4.set_xlabel("Strain")
        self.mplwidget.canvas.axes4.set_ylabel("Stress")
        self.mplwidget.canvas.axes4.scatter(X3, Y3, color='green')
        self.mplwidget.canvas.axes4.plot(X3, slope3 * X3, color='black', label='E=%0.3f(R2 = %0.5f)' % (slope3, a4))
        self.mplwidget.canvas.axes4.legend(loc='upper left')
        self.mplwidget.canvas.axes4.set_title('Split-3')
        self.mplwidget.canvas.draw()
        self.pg.setValue(40)


        dicr = {a1: slope, a2: slope1, a3: slope2, a4: slope3}
        q, w = max(dicr.items(), key=lambda k: k[1])
        global E_M
        E_M = w
        self.mplwidget_3.canvas.axes.clear()
        ref_strain = 0.002 + X
        ref_stress = E_M * X
        x_new = np.array(X)
        y_new = np.array(Y)
        x_ref_strin = np.array(ref_strain)
        y_ref_stress = np.array(ref_stress)
        i1 = pd.DataFrame(ref_strain)
        i2 = pd.DataFrame(ref_stress)
        i3 = pd.DataFrame(x_new)
        i4 = pd.DataFrame(y_new)
        line_1 = LineString(np.column_stack((i1, i2)))
        line_2 = LineString(np.column_stack((i3, i4)))
        intersec = line_1.intersection(line_2)
        j1, j2 = intersec.xy
        global v
        global g
        v = j1[0]
        g = j2[0]

        def find_nearst(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            jux = np.where(array == array[idx])
            return array[idx], jux

        idc, yidx = find_nearst(x_new, v)
        mica = int(yidx[0])

        us = max(Y)
        cs_x = X[Y.argmax()]
        maxindx = np.where(X == cs_x)
        ul = int(maxindx[0])  # max index

        # plt.plot(ref_strain,ref_stress)
        X_y = v
        Y_y = g
        X_u = cs_x
        Y_u = us
        global Tangent
        Tangent = (Y_u - Y_y) / (X_u - X_y)


        point1 = [v,g]
        point2 = [cs_x, us]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        self.mplwidget_3.canvas.axes.plot(X, Y)
        self.mplwidget_3.canvas.axes.scatter(v,g, color='limegreen',label='Yield Strength =%0.3f ' % g)
        self.mplwidget_3.canvas.axes.scatter(cs_x, us, color='red', label='Ultimate Strength =%0.3f ' % us)
        self.mplwidget_3.canvas.axes.plot(x_values, y_values, color='blue', label='Tangnet Modulo =%0.3f' % Tangent)
        self.mplwidget_3.canvas.axes.legend(loc='lower right')
        self.mplwidget_3.canvas.axes.set_xlim(0, max(X)+0.020)
        self.mplwidget_3.canvas.axes.set_ylim(0, max(Y) + 10)
        self.mplwidget_3.canvas.axes.set_title('E=%0.3f' % E_M)
        self.mplwidget_3.canvas.axes.set_xlabel("Strain")
        self.mplwidget_3.canvas.axes.set_ylabel("Stress")
        self.mplwidget_3.canvas.draw()
        self.pg.setValue(50)
        mul = all_data[mica:ul].reset_index()
        pls_x = mul['Strain'].to_numpy()
        pls_y = mul['Stress'].to_numpy()
        pls_xmin = min(pls_x)
        pls_ymin=min(pls_y)
        pls_xnew = pls_x - (pls_y / w)
        necsv = pd.DataFrame([pls_xnew, pls_y])
        zxd = necsv.T
        dax = []
        dax.insert(0, {0: 0,1: g})
        global csg
        csg = pd.concat([pd.DataFrame(dax), zxd], ignore_index=True)
        global cix,ciy
        cix = csg[0].to_numpy()
        ciy = csg[1].to_numpy()
        self.mplwidget_4.canvas.axes.clear()
        self.mplwidget_4.canvas.axes.plot(cix,ciy)
        self.mplwidget_4.canvas.axes.set_title("Multilinear Isotropic Hardening")
        self.mplwidget_4.canvas.axes.set_xlabel('Plastic Strain')
        self.mplwidget_4.canvas.axes.set_ylabel('Stress')
        self.mplwidget_4.canvas.axes.set_xlim(0, max(pls_xnew))
        self.mplwidget_4.canvas.axes.set_ylim(min(pls_y), max(pls_y))
        self.mplwidget_4.canvas.draw()
        self.expo.setEnabled(True)
        self.pg.setValue(70)

        def powerlaw(x, A, n):
            return A * x ** n

        slopec, interceptc = curve_fit(powerlaw,pls_xnew,pls_y,maxfev=80000)
        global Q1
        Q1=slopec[0]
        global Q2
        Q2=slopec[1]
        self.mplwidget_5.canvas.axes.clear()
        self.mplwidget_5.canvas.axes.scatter(cix, ciy,color='orange')
        self.mplwidget_5.canvas.axes.plot(cix, slopec[0] * cix ** slopec[1], label='K =%0.3f n =%0.5f' % (slopec[0], slopec[1]))
        self.mplwidget_5.canvas.axes.legend(loc='lower right')
        self.mplwidget_5.canvas.axes.set_xlabel('Strain')
        self.mplwidget_5.canvas.axes.set_ylabel('Stress')
        self.mplwidget_5.canvas.draw()
        self.pg.setValue(80)

        def voce(x, k, R, T, b):
            return k + R * x + T * (1 - np.exp(-b * x))

        maxfe3 = 80000
        p, c = curve_fit(voce, cix, ciy, maxfev=maxfe3)
        global r1,r2,r3,r4
        r1=p[0]
        r2=p[1]
        r3=p[2]
        r4=p[3]
        self.mplwidget_6.canvas.axes.clear()
        self.mplwidget_6.canvas.axes.scatter(cix,ciy,color='orange')
        self.mplwidget_6.canvas.axes.plot(cix, p[0] + p[1] * cix+ p[2] * (1 - np.exp(-p[3] * cix)),label='k =%0.3f R0=%0.5f Rinf =%0.5f b= -%0.5f' % (p[0], p[1], p[2], p[3]))
        self.mplwidget_6.canvas.axes.legend(loc='lower right')
        self.mplwidget_6.canvas.axes.set_xlabel('Plastic Strain')
        self.mplwidget_6.canvas.axes.set_ylabel('Stress')
        self.mplwidget_6.canvas.draw()
        self.tabWidget_2.setTabEnabled(2, True)
        self.pg.setValue(100)
        self.tabWidget.setTabVisible(7, True)
        self.tabWidget_4.setTabVisible(0,True)
        self.tabWidget_4.setTabVisible(1, True)
        self.tabWidget_4.setTabVisible(2, True)

    def expot(self):
        name = QFileDialog.getSaveFileName(self, 'Save File', filter='*.csv')
        if (name[0] == ''):
            pass
        else:
            csg.to_csv(name[0], index=False,header=['Plastic Strain', 'Stress'])

    def evalx(self):
       try:
        self.tabWidget_3.setTabVisible(0, False)
        self.tabWidget_3.setTabVisible(1, True)
        def yeilds(data):
         cipa=(data['Strain'] == 0).all()
         if (cipa==False):
            def linear(X, a):
                return a * X
            def chi(X, Y, p1):
                residuals = Y - linear(X, *p1)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((Y3 - np.mean(Y3)) ** 2)
                score = 1 - (ss_res / ss_tot)
                return score

            maxfe = 100000
            X = data['Strain'].to_numpy()
            Y = data['Stress'].to_numpy()
            halfdf = len(data) // 2
            first_half = data.iloc[:halfdf]
            secondhalf = len(first_half) // 2
            second_half = data.iloc[:secondhalf]
            thirdhalf = len(second_half) // 2
            third_half = data.iloc[:thirdhalf]

            X1 = first_half['Strain'].to_numpy()
            Y1 = first_half['Stress'].to_numpy()
            X2 = second_half['Strain'].to_numpy()
            Y2 = second_half['Stress'].to_numpy()
            X3 = third_half['Strain'].to_numpy()
            Y3 = third_half['Stress'].to_numpy()

            slope, intercept = curve_fit(linear, X, Y, maxfev=maxfe)
            slope1, intercept1 = curve_fit(linear, X1, Y1, maxfev=maxfe)
            slope2, intercept2 = curve_fit(linear, X2, Y2, maxfev=maxfe)
            slope3, intercept3 = curve_fit(linear, X3, Y3, maxfev=maxfe)

            a1 = chi(X, Y, slope)
            a2 = chi(X1, Y1, slope1)
            a3 = chi(X2, Y2, slope3)
            a4 = chi(X3, Y3, slope3)
            dicr = {a1: slope, a2: slope1, a3: slope2, a4: slope3}
            q, w = max(dicr.items(), key=lambda k: k[1])
            E_M = w
            ref_strain = 0.002 + X
            ref_stress = E_M * X
            x_new = np.array(X)
            y_new = np.array(Y)
            i1 = pd.DataFrame(ref_strain)
            i2 = pd.DataFrame(ref_stress)
            i3 = pd.DataFrame(x_new)
            i4 = pd.DataFrame(y_new)
            line_1 = LineString(np.column_stack((i1, i2)))
            line_2 = LineString(np.column_stack((i3, i4)))
            intersec = line_1.intersection(line_2)
            j1, j2 = intersec.xy
            v = j1[0]
            g = j2[0]

            return w, g;
         else:
              w=0
              g=0
              return w, g;
        def cows(x, C, P):
            return 1 + (x / C) ** (1 / P)

        global strate
        self.pg.setValue(0)
        c1 = self.ref1.text()
        c2 = self.ref2.text()
        c3 = self.ref3.text()
        c4 = self.ref4.text()
        c5 = self.ref5.text()
        c6 = self.ref6.text()

        self.pg.setValue(10)
        E1,YST=yeilds(all_data1)
        self.pg.setValue(20)
        E2,Y2=yeilds(all_data2)
        self.pg.setValue(30)
        E3,Y3=yeilds(all_data3)
        self.pg.setValue(40)
        E4,Y4=yeilds(all_data4)
        self.pg.setValue(50)
        E5,Y5=yeilds(all_data5)
        self.pg.setValue(60)
        E5,Y6=yeilds(all_data6)
        self.pg.setValue(70)
        ud1=Y2/YST
        ud2=Y3/YST
        ud3=Y4/YST
        ud4=Y5/YST
        ud5=Y6/YST
        k = [float(c2), float(c3), float(c4), float(c5), float(c6)]
        l=[ud1,ud2,ud3,ud4,ud5]
        dj=[[float(c2),ud1],[float(c3),ud2],[float(c4),ud3],[float(c5),ud4],[float(c6),ud5]]
        strate = pd.DataFrame(dj, columns=['Strain_rate','Stress_ratio'])
        global k1,l1,na
        na = strate[(strate.T != 0).any()]
        k1= na['Strain_rate'].to_numpy()
        l1= na['Stress_ratio'].to_numpy()
        moxc=pandasModel(na)
        self.dym.setModel(moxc)
        z1, z2 = curve_fit(cows, k1,l1,maxfev=90000)
        global C
        global P
        C=z1[0]
        P=z1[1]
        l22=1 + (k1 / z1[0]) ** (1 / z1[1])
        self.pg.setValue(80)
        self.mplwidget_7.canvas.axes.clear()
        self.mplwidget_7.canvas.axes.set_xlabel("Strain Rate")
        self.mplwidget_7.canvas.axes.set_ylabel("Stress Ratio")
        self.mplwidget_7.canvas.axes.scatter(k1, l1, color='orange',label='Dynamic Experients')
        self.mplwidget_7.canvas.axes.plot(k1,l22,color='blue',label='Fit parameters C= %0.3f P= %0.3f'%(z1[0],z1[1]))
        self.mplwidget_7.canvas.axes.legend(loc='upper left')
        self.mplwidget_7.canvas.axes.set_title('Cowper Symods')
        self.mplwidget_7.canvas.draw()
        self.tabWidget_2.setTabEnabled(2, True)
        self.pg.setValue(100)
       except ZeroDivisionError:
           self.pg.setValue(0)
           msg = QMessageBox()
           msg.setIcon(QMessageBox.Warning)
           msg.setWindowTitle("Warning")
           msg.setText("Input files may contains all Nulls or No Input Files are Provided")

           def msgButtonClick(i):
               msg.close()

           msg.setStandardButtons(QMessageBox.Cancel)
           msg.show()
           msg.buttonClicked.connect(msgButtonClick)
       self.tabWidget.setTabVisible(7, True)

    def zenza(self):
        self.pg.setValue(0)

        def yep(data):
         try:
            def linear(X, a):
                return a * X

            def rsq(Ypre, Y):
                y1 = sum(pow(Ypre, 2))
                y2 = len(Y) * np.var(Y)
                return 1 - (y1 / y2)

            def chi(X, Y, p1):
                residuals = Y - linear(X, *p1)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((Y3 - np.mean(Y3)) ** 2)
                score = 1 - (ss_res / ss_tot)
                return score
            X = data['Strain'].to_numpy()
            Y = data['Stress'].to_numpy()

            halfdf = len(data) // 2
            first_half = data.iloc[:halfdf]
            secondhalf = len(first_half) // 2
            second_half = data.iloc[:secondhalf]
            thirdhalf = len(second_half) // 2
            third_half = data.iloc[:thirdhalf]

            X1 = first_half['Strain'].to_numpy()
            Y1 = first_half['Stress'].to_numpy()
            X2 = second_half['Strain'].to_numpy()
            Y2 = second_half['Stress'].to_numpy()
            X3 = third_half['Strain'].to_numpy()
            Y3 = third_half['Stress'].to_numpy()

            maxfe = 100000

            slope, intercept = curve_fit(linear, X, Y, maxfev=maxfe)
            slope1, intercept1 = curve_fit(linear, X1, Y1, maxfev=maxfe)
            slope2, intercept2 = curve_fit(linear, X2, Y2, maxfev=maxfe)
            slope3, intercept3 = curve_fit(linear, X3, Y3, maxfev=maxfe)

            a1 = chi(X, Y, slope)
            a2 = chi(X1, Y1, slope1)
            a3 = chi(X2, Y2, slope3)
            a4 = chi(X3, Y3, slope3)



            dicr = {a1: slope, a2: slope1, a3: slope2, a4: slope3}
            global wp
            q, wp = max(dicr.items(), key=lambda k: k[1])
            E_M = wp

            ref_strain = 0.002 + X
            ref_stress = E_M * X
            x_new = np.array(X)
            y_new = np.array(Y)
            slope, intercept = linear_regression(ref_strain, ref_stress)
            slope2, intercept2 = linear_regression(X, ref_stress)
            ks = np.polyfit(x_new, y_new, deg=15)
            yj = E_M * X + intercept
            cp = np.polyval(ks, x_new)
            cp_match = np.array(cp)
            y_match = np.array(yj)
            idx = np.argwhere(np.diff(np.sign(yj - cp_match))).flatten()
            x_inc = X[idx]
            y_inc = cp_match[idx]
            mapas = float(x_inc)
            yiel = slope2 * mapas + intercept2
            x_new = np.array(X)
            y_new = np.array(Y)
            x_ref_strin = np.array(ref_strain)
            y_ref_stress = np.array(ref_stress)
            i1 = pd.DataFrame(ref_strain)
            i2 = pd.DataFrame(ref_stress)
            i3 = pd.DataFrame(x_new)
            i4 = pd.DataFrame(y_new)
            line_1 = LineString(np.column_stack((i1, i2)))
            line_2 = LineString(np.column_stack((i3, i4)))
            intersec = line_1.intersection(line_2)
            j1, j2 = intersec.xy
            v = j1[0]
            g = j2[0]

            us = max(Y)
            cs_x = X[Y.argmax()]
            maxindx = np.where(X == cs_x)
            ul = int(maxindx[0])  # max index

            X_y = v
            Y_y = g
            X_u = cs_x
            Y_u = us
            global Tangent
            Tangent = (Y_u - Y_y) / (X_u - X_y)

            return wp, g, Tangent;
         except:
             return 0,0,0
        def plsti(data):
         try:
            ds = data
            X = ds['Strain'].to_numpy()
            Y = ds['Stress'].to_numpy()

            ref_strain = 0.002 + X
            ref_stress = wp* X
            x_new = np.array(X)
            y_new = np.array(Y)
            x_ref_strin = np.array(ref_strain)
            y_ref_stress = np.array(ref_stress)
            i1 = pd.DataFrame(ref_strain)
            i2 = pd.DataFrame(ref_stress)
            i3 = pd.DataFrame(x_new)
            i4 = pd.DataFrame(y_new)
            line_1 = LineString(np.column_stack((i1, i2)))
            line_2 = LineString(np.column_stack((i3, i4)))
            intersec = line_1.intersection(line_2)
            j1, j2 = intersec.xy
            v = j1[0]
            g = j2[0]

            def find_nearst(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                jux = np.where(array == array[idx])
                return array[idx], jux

            idc, yidx = find_nearst(x_new, v)
            mica = int(yidx[0])

            us = max(Y)
            cs_x = X[Y.argmax()]
            maxindx = np.where(X == cs_x)
            ul = int(maxindx[0])  # max index


            mul = data[mica:ul].reset_index()
            pls_x = mul['Strain'].to_numpy()
            pls_y = mul['Stress'].to_numpy()

            pls_xnew = pls_x - (pls_y / wp)
            necsv = pd.DataFrame([pls_xnew, pls_y])
            zxd = necsv.T
            dax = []
            dax.insert(0, {0: 0, 1: g})
            global plza
            plza = pd.concat([pd.DataFrame(dax), zxd], ignore_index=True)
            return plza
         except:
            emp = pd.DataFrame(np.zeros((1, 2)))
            return emp
        def plstic(data):
         try:
            def powerlaw(x, A, n):
                return A * x ** n
            ds = data
            X = ds['Strain'].to_numpy()
            Y = ds['Stress'].to_numpy()

            ref_strain = 0.002 + X
            ref_stress = wp* X
            x_new = np.array(X)
            y_new = np.array(Y)
            x_ref_strin = np.array(ref_strain)
            y_ref_stress = np.array(ref_stress)
            i1 = pd.DataFrame(ref_strain)
            i2 = pd.DataFrame(ref_stress)
            i3 = pd.DataFrame(x_new)
            i4 = pd.DataFrame(y_new)
            line_1 = LineString(np.column_stack((i1, i2)))
            line_2 = LineString(np.column_stack((i3, i4)))
            intersec = line_1.intersection(line_2)
            j1, j2 = intersec.xy
            v = j1[0]
            g = j2[0]

            def find_nearst(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                jux = np.where(array == array[idx])
                return array[idx], jux

            idc, yidx = find_nearst(x_new, v)
            mica = int(yidx[0])

            us = max(Y)
            cs_x = X[Y.argmax()]
            maxindx = np.where(X == cs_x)
            ul = int(maxindx[0])  # max index


            mul = data[mica:ul].reset_index()
            pls_x = mul['Strain'].to_numpy()
            pls_y = mul['Stress'].to_numpy()

            pls_xnew = pls_x - (pls_y / wp)
            necsv = pd.DataFrame([pls_xnew, pls_y])
            slopec, interceptc = curve_fit(powerlaw, pls_xnew, pls_y, maxfev=80000)
            Q1 = slopec[0]
            Q2 = slopec[1]
            return Q1,Q2
         except:

            return 0,0

        C_CAL = C
        P_CAL = P

        dep = self.den.text()
        mu = self.nux.text()
        deca = float(dep)
        nux = float(mu)

        ref_e=all_data1
        ref_e1=all_data2
        ref_e2 = all_data3
        ref_e3 = all_data4
        ref_e4 = all_data5
        ref_e5 = all_data6

        z11=yep(ref_e)
        z22 = yep(ref_e1)
        z33 = yep(ref_e2)
        z44 = yep(ref_e3)
        z55 = yep(ref_e4)
        z66 = yep(ref_e5)


        ref_E=float(z11[0])
        ref_yield=float(z11[1])
        ref_tan=float(z11[2])

        ref_yield1 = float(z22[1])
        ref_tan1 = float(z22[2])

        ref_yield2 = float(z33[1])
        ref_tan2 = float(z33[2])

        ref_yield3 = float(z44[1])
        ref_tan3 = float(z44[2])

        ref_yield4 = float(z55[1])
        ref_tan4 = float(z55[2])

        ref_yield5 = float(z66[1])
        ref_tan5 = float(z66[2])
        self.pg.setValue(30)

        t1 = plsti(ref_e1)
        t2 = plsti(ref_e2)
        t3 = plsti(ref_e3)
        t4 = plsti(ref_e4)
        t5 = plsti(ref_e5)
        self.pg.setValue(50)

        c1 = self.ref1.text()
        c2 = self.ref2.text()
        c3 = self.ref3.text()
        c4 = self.ref4.text()
        c5 = self.ref5.text()
        c6 = self.ref6.text()

        dj1 = [[float(c2), 1], [float(c3), 2], [float(c4), 3], [float(c5), 4], [float(c6), 5]]
        strate1 = pd.DataFrame(dj1)
        nazw=strate1
        naz=nazw[(nazw!= 0).all(1)]
        self.sia.append('*KEYWORD')
        self.sia.append('*MAT_PLASTIC_KINEMATIC_TITLE')
        self.sia.append('Sample')
        self.sia.append('1,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,0' % (deca, ref_E, nux, ref_yield, ref_tan))
        self.sia.append('%0.2f,%0.2f,0.0,0.0'%(C_CAL,P_CAL))
        self.sia.append('*END')

        self.lita.append('*KEYWORD')
        self.lita.append('*MAT_PIECEWISE_LINEAR_PLASTICITY_TITLE')
        self.lita.append('Sample')
        self.lita.append('1,%0.2f,%0.2f,%0.2f,%0.2f,0,1E+20,0' % (deca, ref_E, nux, ref_yield))
        self.lita.append('%0.2f,%0.2f,1,0,0' % (C_CAL, P_CAL))
        self.lita.append('0,0,0,0,0,0,0,0')
        self.lita.append('0,0,0,0,0,0,0,0')
        self.lita.append('*DEFINE_CURVE')
        self.lita.append('101,0,1,1,0,0,0,0')
        for i, r in t1.iterrows():
            self.lita.append("%0.3f,%0.3f" % (r[0], r[1]))
        self.lita.append('*DEFINE_CURVE')
        self.lita.append('102,0,1,1,0,0,0,0')
        for i, r in t2.iterrows():
            self.lita.append("%0.3f,%0.3f" % (r[0], r[1]))
        self.lita.append('*DEFINE_CURVE')
        self.lita.append('103,0,1,1,0,0,0,0')
        for i, r in t3.iterrows():
            self.lita.append("%0.3f,%0.3f" % (r[0], r[1]))
        self.lita.append('*DEFINE_CURVE')
        self.lita.append('104,0,1,1,0,0,0,0')
        for i, r in t4.iterrows():
            self.lita.append("%0.3f,%0.3f" % (r[0], r[1]))
        self.lita.append('*DEFINE_CURVE')
        self.lita.append('105,0,1,1,0,0,0,0')
        for i, r in t5.iterrows():
            self.lita.append("%0.3f,%0.3f" % (r[0], r[1]))
        self.lita.append('*DEFINE_TABLE_TITLE')
        self.lita.append('strain_rate_vs_yield_table')
        self.lita.append('1001,0,0')
        for i, r in naz.iterrows():
            self.lita.append("%0.1f,%0.1f" % (r[0], r[1]))
        self.lita.append('*END')
        tusa=plstic(ref_e)
        DX1=tusa[0]
        DX2=tusa[1]
        self.pg.setValue(70)
        self.kia.append('*KEYWORD')
        self.kia.append('*MAT_POWER_LAW_PLASTICITY_TITLE')
        self.kia.append('Sample')
        self.kia.append('1,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f' % (deca, ref_E, nux, DX1, DX2,C_CAL,P_CAL))
        self.kia.append('%0.2f,0,0' % ref_yield)
        self.kia.append('*END')


        ydj1 = [[float(c1), ref_yield],[float(c2), ref_yield1], [float(c3), ref_yield2], [float(c4), ref_yield3], [float(c5), ref_yield4], [float(c6), ref_yield5]]
        ystrate1 = pd.DataFrame(ydj1)
        ynazw = ystrate1
        ynaz = ynazw[(ynazw != 0).all(1)]


        tdj1 = [[float(c1), ref_tan], [float(c2), ref_tan1], [float(c3), ref_tan2], [float(c4), ref_tan3],
                [float(c5), ref_tan4], [float(c6), ref_tan5]]
        tstrate1 = pd.DataFrame(tdj1)
        tnazw = tstrate1
        tnaz = tnazw[(tnazw != 0).all(1)]
        self.pg.setValue(80)
        self.lita_2.append('*KEYWORD')
        self.lita_2.append('*MAT_STRAIN_RATE_DEPENDENT_PLASTICITY_TITLE')
        self.lita_2.append('samp')
        self.lita_2.append('1,%0.3f,%0.3f,%0.3f,0'%(deca,ref_E,nux))
        self.lita_2.append('101,%0.3f,0,102,0,0'%ref_tan)
        self.lita_2.append('*DEFINE_CURVE')
        self.lita_2.append('101,0,1,1,0,0,0,0')
        for i, r in ynaz.iterrows():
            self.lita_2.append("%0.3f,%0.3f" % (r[0], r[1]))
        self.lita_2.append('*DEFINE_CURVE')
        self.lita_2.append('102,0,1,1,0,0,0,0')
        for i, r in tnaz.iterrows():
            self.lita_2.append("%0.3f,%0.3f" % (r[0], r[1]))
        self.lita_2.append('*END')
        self.pg.setValue(100)

    def rep(self):
       try:
        dep=self.den.text()
        mu=self.nux.text()
        E=float(E_M)
        y=g
        ta=Tangent
        self.biso1.append('/prep7')
        deca=float(dep)
        nux=float(mu)
        C=Q1
        P=Q2
        kk=r1
        RR=r2
        Tt=r3
        bb=r4
        lexc=len(csg.index)
        self.biso1.append('MP,DENS,1,%0.3f' %deca)
        self.biso1.append('MP,EX,1,%0.3f'%E)
        self.biso1.append('MP,NUXY,1,%0.3f'%nux)
        self.biso1.append('TB,BISO,1,1')
        self.biso1.append('TBDATA,1,%0.3f,%0.3f'%(y,ta))
        self.POW1.append('/prep7')
        self.POW1.append('MP,DENS,1,%0.3f' % deca)
        self.POW1.append('MP,EX,1,%0.3f' % E)
        self.POW1.append('MP,NUXY,1,%0.3f' % nux)
        self.POW1.append('TB,NLISO,1,1,POWER')
        self.POW1.append('TBDATA,1,%0.3f,%0.3f' % (C,P))
        self.VOC1.append('/prep7')
        self.VOC1.append('MP,DENS,1,%0.3f' % deca)
        self.VOC1.append('MP,EX,1,%0.3f' % E)
        self.VOC1.append('MP,NUXY,1,%0.3f' % nux)
        self.VOC1.append('TB,NLISO,1,1,,VOCE')
        self.VOC1.append('TBDATA,1,%0.3f,%0.3f,%0.3f,%0.3f' % (kk,RR,Tt,bb))
        self.MISO1.append('/prep7')
        self.MISO1.append('MP,DENS,1,%0.3f'%deca)
        self.MISO1.append('MP,EX,1,%0.3f'%E)
        self.MISO1.append('MP,NUXY,1,%0.3f'%nux)
        self.MISO1.append('TB,PLAS,1,1,%0.3f,MISO'%lexc)
        self.MISO1.append('TBTEMP,25')
        for i, r in csg.iterrows():
            self.MISO1.append("TBPT,,%0.3f ,%0.3f" % (r[0], r[1]))
        self.sia.append('*KEYWORD')
        self.sia.append('*MAT_PLASTIC_KINEMATIC_TITLE')
        self.sia.append('Sample')
        self.sia.append('1,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,0'%(deca,E,nux,y,ta))
        self.sia.append('0.0,0.0,0.0,0.0')
        self.sia.append('*END')
        self.kia.append('*KEYWORD')
        self.kia.append('*MAT_POWER_LAW_PLASTICITY_TITLE')
        self.kia.append('Sample')
        self.kia.append('1,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,0,0'%(deca,E,nux,C,P))
        self.kia.append('%0.2f,0,0' %y)
        self.kia.append('*END')
        self.lita.append('*KEYWORD')
        self.lita.append('*MAT_PIECEWISE_LINEAR_PLASTICITY_TITLE')
        self.lita.append('Sample')
        self.lita.append('1,%0.2f,%0.2f,%0.2f,%0.2f,0,1E+20,0' % (deca, E, nux,y))
        self.lita.append('0,0,1,0,0')
        self.lita.append('0,0,0,0,0,0,0,0')
        self.lita.append('0,0,0,0,0,0,0,0')
        self.lita.append('*DEFINE_CURVE')
        self.lita.append('1,0,1,1,0,0,0,0')
        for i, r in csg.iterrows():
             self.lita.append("%0.3f,%0.3f" % (r[0],r[1]))
        # self.lita.append('*DEFINE_CURVE')
        # self.lita.append('2,0,1,1,0,0,0,0')
        # for i, r in na.iterrows():
        #     self.lita.append("%0.3f ,%0.3f" % (r['Strain_rate'], r['Dynamic Stress']))
        self.lita.append('*END')
        self.biso1_2.append('mapdl.mp("EX" ,1,%0.3f)'%E)
        self.biso1_2.append('mapdl.mp("DENS",1,%0.3f)'%deca)
        self.biso1_2.append('mapdl.mp("NUXY",1,%0.3f)' % nux)
        self.biso1_2.append('mapdl.tb("BISO",1,1)')
        self.biso1_2.append('mapdl.tbdata(1,%0.3f,%0.3f)'%(y,ta))
       except:
           msg = QMessageBox()
           msg.setIcon(QMessageBox.Warning)
           msg.setWindowTitle("Warning")
           msg.setText("Calibrate first strain to gather Elasticity terms")

           def msgButtonClick(i):
               msg.close()

           msg.setStandardButtons(QMessageBox.Cancel)
           msg.show()
           msg.buttonClicked.connect(msgButtonClick)

class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # QtPy
    dark_stylesheet = qdarkstyle.load_stylesheet()
    # PyQtGraph
    #dark_stylesheet = qdarkstyle.load_stylesheet(qt_api=os.environ('PYQTGRAPH_QT_LIB'))
    # Qt.Py
    #dark_stylesheet = qdarkstyle.load_stylesheet(qt_api=Qt.__binding__)
    app.setStyleSheet(dark_stylesheet)
    app.setStyle('fusion')
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())