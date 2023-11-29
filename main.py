import kivy
from kivy.app import App
from kivy.uix.label import Label
import cv2
import numpy as np
from multiprocessing import Process
  
class MyLabelApp(App):
    def build(self):
        caption = cv2.VideoCapture(0,cv2.CAP_V4L2)
        if not caption.isOpened():
            lbl2 = Label(text ="cannot open camera 0")
            return lbl2
            caption.release()
        else:
            caption.release()
        caption1 = cv2.VideoCapture(1,cv2.CAP_V4L2)
        if not caption1.isOpened():
            lbl3 = Label(text ="cannot open camera 1")
            return lbl3
            caption1.release()
        else:
            caption1.release()
        lbl = Label(text ="Android Two Cam Is running, Close window and save ouput.avi locally!!! ")
        return lbl
def cap():
    cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
    cap1 = cv2.VideoCapture(1,cv2.CAP_V4L2)
    out = cv2.VideoWriter('/storage/emulated/0/dcim/output.avi',cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS), (1440,720))
    while(cap.isOpened()):
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()
        if ret == True:
            reframe = cv2.resize(frame,(720, 720), interpolation = cv2.INTER_AREA)
            reframe1 = cv2.resize(frame1,(720, 720), interpolation = cv2.INTER_AREA)
            both = np.column_stack((reframe, reframe1))
            out.write(both) 
if __name__ == "__main__":
    p1 = Process(target=cap)
    p1.start()
    label = MyLabelApp()
    label.run()
    label.on_stop(p1.terminate())