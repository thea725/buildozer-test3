from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
from kivy.clock import Clock

from os import environ
from sys import platform as _sys_platform

pixelsPerMetric = 27.9394
def platform():
    if "ANDROID_ARGUMENT" in environ:
        return "android"
    elif _sys_platform in ('win32', 'cygwin'):
        return "win"

class VideoApp(App):
    def build(self):
        if platform() == "android":
            path = '/data/data/org.test.recycleai/files/app/vid4.mp4'
        elif platform() == "win":
            path = "vid4.mp4"
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        self.container = BoxLayout(orientation='vertical')
        self.image = Image()
        self.container.add_widget(self.image)

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update every 1/30th of a second

        return self.container

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            print("End of video.")
            return

        # normal = self.defisheye(frame)
        # enhance = self.normalization(normal)
        # result = self.edge_detection(normal, enhance)

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame_texture = self.texture_from_frame(frame)
        self.image.texture = frame_texture
    
    def texture_from_frame(self, frame):
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

if __name__ == '__main__':
    VideoApp().run()