from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from kivy.clock import Clock

from os import environ
from sys import platform as _sys_platform

def platform():
    if "ANDROID_ARGUMENT" in environ:
        return "android"
    elif _sys_platform in ('win32', 'cygwin'):
        return "win"

def normalization(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(20, 20))
    enhanced_image = clahe.apply(normalized_image)
    equ = cv2.equalizeHist(enhanced_image)
    
    upper_black = 75
    equ[np.where(equ <= upper_black)] = 0
    equ[np.where(equ <= upper_black)] = 0

    return equ

class ImageApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.layout.add_widget(self.image)

        if platform() == "android":
            path = "/data/data/org.test.recycleai/files/app/vid4.mp4"
        elif platform() == "win":
            path = "vid4.mp4"
        self.cap = cv2.VideoCapture(path)

        # Atur pemanggilan fungsi update setiap 1/30 detik
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return self.layout

    def update(self, *args):
        # Baca gambar menggunakan OpenCV
        ret, frame = self.cap.read()
        if not ret:
            return
        # normal = defisheye(frame)
        result = normalization(frame)
        # result = edge_detection(normal, enhance)
        
        # Ubah warna dari BGR ke RGB
        frame_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Buat tekstur Kivy dari citra OpenCV
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(cv2.flip(frame_rgb, 0).tobytes(), colorfmt='rgb', bufferfmt='ubyte')

        # Tampilkan gambar di aplikasi
        self.image.texture = texture

if __name__ == '__main__':
    ImageApp().run()