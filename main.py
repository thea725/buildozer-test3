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

def defisheye(img):
    DIM = img.shape[:2][::-1]
    balance = 0
    K = np.array([[1122.4054962744387, 0.0, 1006.1145835723129], [0.0, 1129.0933478170655, 527.4670240270237], [0.0, 0.0, 1.0]])
    D = np.array([[-0.21503184621950375], [0.7441653867540186], [-1.3654196840660953], [0.8759864071569387]])
    dim1 = [1920, 1080]
    dim2 = [1920, 1080]
    dim3 = [1920, 1080]

    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance, fov_scale=1)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

    dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return dst
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

        # Ubah warna dari BGR ke RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        normal = defisheye(frame_rgb)

        # Buat tekstur Kivy dari citra OpenCV
        texture = Texture.create(size=(normal.shape[1], normal.shape[0]), colorfmt='rgb')
        texture.blit_buffer(cv2.flip(normal, 0).tostring(), colorfmt='rgb', bufferfmt='ubyte')

        # Tampilkan gambar di aplikasi
        self.image.texture = texture

if __name__ == '__main__':
    ImageApp().run()
