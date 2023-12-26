from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
from kivy.clock import Clock
from ultralytics import YOLO

import os
from sys import platform as _sys_platform

model = YOLO(os.path.join(os.getcwd(), 'best.pt'))

class ImageApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image = Image()
        self.layout.add_widget(self.image)

        path = os.path.join(os.getcwd(), 'vid4.mp4')
        self.cap = cv2.VideoCapture(path)

        # Atur pemanggilan fungsi update setiap 1/30 detik
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return self.layout

    def update(self, *args):
        # Baca gambar menggunakan OpenCV
        ret, frame = self.cap.read()
        if not ret:
            return

        # Ubah warna dari BGR ke RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame_rgb)
        results = results[0].plot()

        # Buat tekstur Kivy dari citra OpenCV
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(cv2.flip(results, 0).tostring(), colorfmt='rgb', bufferfmt='ubyte')

        # Tampilkan gambar di aplikasi
        self.image.texture = texture

if __name__ == '__main__':
    ImageApp().run()