import time
from pynput.mouse import Controller

controller = Controller()


for i in range(1000) :
    #controller.position = (i, i)

    controller.move(-1, -1)
    time.sleep(0.01)