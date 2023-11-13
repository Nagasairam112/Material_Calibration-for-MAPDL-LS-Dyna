# ------------------------------------------------- ----- 
# -------------------- mplwidget.py -------------------- 
# -------------------------------------------------- ---- 
from  PyQt5.QtWidgets  import *

from  matplotlib.backends.backend_qt5agg  import  FigureCanvas

from  matplotlib.figure  import  Figure
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as
NavigationToolbar)

    
class  MplWidget4 ( QWidget ):
    
    def  __init__ ( self ,  parent  =  None ):
        QWidget . __init__ ( self ,  parent )
        self . canvas  =  FigureCanvas ( Figure ())
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget((self.toolbar))
        self.canvas.figure.subplots_adjust(left=0.095, right=0.96, top=0.955, bottom=0.070, wspace=0.200, hspace=0.200)
        self . setLayout ( vertical_layout )


