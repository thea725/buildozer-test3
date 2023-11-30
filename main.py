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

def euclidean(point1, point2):
    # Check if the dimensions of the two points are the same
    if len(point1) != len(point2):
        raise ValueError("The two points must have the same number of dimensions")

    # Compute the squared differences for each dimension
    squared_diff = [(point2[i] - point1[i]) ** 2 for i in range(len(point1))]

    # Sum the squared differences and take the square root
    distance = np.sqrt(sum(squared_diff))

    return distance
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
pixelsPerMetric = 27.9394

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
def edge_detection(frame, img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 10  # Adjust this value as needed
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]

    orig = frame.copy()
    for c in filtered_contours:
        # Measure the size of the object (contour) and calculate the scale factor
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = order_points(box)
        
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        # compute the Euclidean distance between the midpoints
        dA = euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = euclidean((tlblX, tlblY), (trbrX, trbrY))

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # if (dimA>5 and dimB>14) and (dimA<15 and dimB<37): #ukuran botol di antara 5-5 dan 14-37cm
        if dimA>1 and dimB>10 and (int(dimB/dimA) < 4 and int(dimB/dimA) > 2):
        # if True:
            # print(dB)
            # print("-",dimB)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            # draw the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 2)

            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}cm".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.putText(orig, "{:.1f}cm".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
    return orig


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
        normal = defisheye(frame)
        enhance = normalization(normal)
        result = edge_detection(normal, enhance)
        
        # Ubah warna dari BGR ke RGB
        frame_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Buat tekstur Kivy dari citra OpenCV
        texture = Texture.create(size=(normal.shape[1], normal.shape[0]), colorfmt='rgb')
        texture.blit_buffer(cv2.flip(frame_rgb, 0).tostring(), colorfmt='rgb', bufferfmt='ubyte')

        # Tampilkan gambar di aplikasi
        self.image.texture = texture

if __name__ == '__main__':
    ImageApp().run()
