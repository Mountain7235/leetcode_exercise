import os
import sys
import time
import traceback
from PySide2 import QtWidgets

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        """Main window, holding all user interface including.
        Args:
          parent: parent class of main window
        Returns:
          None
        Raises:
          None
        """
        super(MainWindow, self).__init__(parent)
        self._width    = 800
        self._height   = 600
        # self._title    = QtWidgets.QLabel('PySide2 is Great', self)
        self._exit_btn = QtWidgets.QPushButton('Exit', self)

        self.setMinimumSize(self._width, self._height)

def error_messages_display():
    cl, exc, tb = sys.exc_info()
    for lastCallStack in traceback.extract_tb(tb):
        errMessage = (f'\n######################## Error Message #############################\n'
                      f'    Error class        : {cl}\n'
                      f'    Error info         : {exc}\n'
                      f'    Error fileName     : {lastCallStack[0]}\n'
                      f'    Error fileLine     : {lastCallStack[1]}\n'
                      f'    Error fileFunction : {lastCallStack[2]}')
        print(errMessage)

if __name__ == '__main__':
    try:
        app = QtWidgets.QApplication(sys.argv)
        w = MainWindow()

        w.show()

        ret = app.exec_()

        sys.exit(ret)
    except:
        error_messages_display()