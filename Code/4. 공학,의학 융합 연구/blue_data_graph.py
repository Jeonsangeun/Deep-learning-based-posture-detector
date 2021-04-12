import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvas as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#
# class MyApp(QWidget):
#
#   def __init__(self):
#       super().__init__()
#       self.initUI()
#
#       self.setLayout(self.layout)
#       self.setGeometry(200, 200, 800, 400)
#
#       self.main_widget = QWidget()
#       self.setCentralWidget(self.main_widget)
#
#       canvas = FigureCanvas(Figure(figsize=(4, 3)))
#       vbox = QVBoxLayout(self.main_widget)
#       vbox.addWidget(canvas)
#
#       self.addToolBar(NavigationToolbar(canvas, self))
#
#       self.ax = canvas.figure.subplots()
#       self.ax.plot([0, 1, 2], [1, 5, 3], '-')
#
#       dynamic_canvas = FigureCanvas(Figure(figsize=(4, 3)))
#       vbox.addWidget(dynamic_canvas)
#
#       self.dynamic_ax = dynamic_canvas.figure.subplots()
#       self.timer = dynamic_canvas.new_timer(
#           100, [(self.update_canvas, (), {})])
#       self.timer.start()
#
#       self.setWindowTitle('Matplotlib in PyQt5')
#       self.setGeometry(300, 100, 600, 600)
#       self.show()
#

#   def update_canvas(self):
#       self.dynamic_ax.clear()
#       t = np.linspace(0, 2 * np.pi, 101)
#       self.dynamic_ax.plot(t, np.sin(t + time.time()), color='deeppink')
#       self.dynamic_ax.figure.canvas.draw()
#
# if __name__ == '__main__':
#   app = QApplication(sys.argv)
#   ex = MyApp()
#   sys.exit(app.exec_())
#

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.data = np.randon(100)
        self.initUI()

        self.setLayout(self.layout)
        self.setGeometry(200, 200, 800, 400)

    def initUI(self):

        self.pushButton = QPushButton("DRAW Graph")
        self.pushButton.clicked.connect(self.btnClicked1)

        self.pushButton2 = QPushButton("Save")
        self.pushButton2.clicked.connect(self.btnClicked2)

        self.pushButton3 = QPushButton("End")

        self.fig, _ = plt.subplots()
        self.canvas = FigureCanvas(self.fig)

        # btn layout
        btnLayout = QVBoxLayout()
        btnLayout.addWidget(self.canvas)

        # canvas Layout
        canvasLayout = QVBoxLayout()
        canvasLayout.addWidget(self.pushButton)
        canvasLayout.addWidget(self.pushButton2)
        canvasLayout.addWidget(self.pushButton3)
        canvasLayout.addStretch(1)

        self.layout = QHBoxLayout()
        self.layout.addLayout(btnLayout)
        self.layout.addLayout(canvasLayout)

    def btnClicked1(self):
        self.fig.clear()
        self.canvas_ax = self.canvas.figure.subplots()
        self.timer = self.canvas.new_timer(100, [(self.update_canvas, (), {})])
        self.timer.start()

    def update_canvas(self):
        self.canvas_ax.clear()
        t = np.linspace(0, 2 * np.pi, 101)
        self.canvas_ax.plot(t, np.sin(t + time.time()), color='deeppink')
        self.canvas_ax.figure.canvas.draw()

    def btnClicked2(self):
        x = np.linspace(0, 10, 100)
        y = np.zeros(100)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_ylim([-180, 180])
        ax.plot(x, y)
        self.canvas.draw()

    def btnClicked3(self):
        x = np.linspace(0, 10, 100)
        y = np.zeros(100)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_ylim([-180, 180])
        ax.plot(x, y)
        self.canvas.draw()


# def recvall(sock, n):
#     data = b''
#     while len(data) < n:
#         socket.send('0')
#         packet = sock.recv(n - len(data))
#         if not packet:
#             return None
#         data += packet
#     return data
#
# def decode_axis(data):
#     result = data.decode()
#     aa = np.zeros(3)
#     aa[0] = round(float(result[:6]) - 500, 2)
#     aa[1] = round(float(result[6:12]) - 500, 2)
#     aa[2] = round(float(result[12:]) - 500, 2)
#     return aa
#
# def main():
#     while True:
#         data = np.zeors(100)
#         if (data == ''):
#             print("")
#             break

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())