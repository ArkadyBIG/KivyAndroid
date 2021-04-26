import kivy

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.utils import platform
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics.texture import Texture
import os
import shutil
from utils import Embedder
from kivy.uix.camera import Camera
import numpy as np
from PIL import Image, ImageDraw
import traceback
import sys
from face_detection import detect_faces
from utils import Embedder


def check_camera_permission():
    """
    Android runtime `CAMERA` permission check.
    """
    if platform != 'android':
        return True
    from android.permissions import Permission, check_permission
    permission = Permission.CAMERA
    return check_permission(permission)


def check_request_camera_permission(callback=None):
    """
    Android runtime `CAMERA` permission check & request.
    """
    if platform != 'android':
        return True
    from android.permissions import Permission, request_permissions, check_permission
    permissions = [Permission.READ_EXTERNAL_STORAGE,
                   Permission.WRITE_EXTERNAL_STORAGE, Permission.CAMERA]

    permissions = [p for p in permissions if not check_permission(p)]
    if permissions:
        request_permissions(permissions, callback)
    return True


class MainWindow(Screen):
    pass


face_data = Embedder(os.getcwd() + '/BlackListImages')
class MyCamera(Camera):
    def __init__(self, **kwargs):
        if kivy.platform == 'android':
            self.resolution = 1920 // 2, 1080 // 2
        #self.face_data = Embedder(os.getcwd() + '/BlackListImages')
        super(MyCamera, self).__init__(**kwargs)

    def _camera_loaded(self, *largs):
        if kivy.platform == 'android':
            self.texture = Texture.create(size=self.resolution, colorfmt='rgb')
            self.texture_size = list(self.texture.size)
        else:
            self.texture = self._camera.texture
            self.texture_size = list(self.texture.size)

    def on_tex(self, *l):
        w, h = self._camera._resolution
        frame = np.frombuffer(self._camera.texture.pixels,
                              np.uint8).reshape(h, w, 4)[..., :3].copy()

        frame = self.process_frame(frame)

        self.put_frame(frame)
        super(MyCamera, self).on_tex(*l)

    def process_frame(self, frame):
        try:
            # emb =
            label = self.parent.children[0].children[1]
            face_found, (name, score) = data = face_data.find_person(
                frame)
            if not face_found:
                color = [0.1, 0.1, 0.1, 1]
                text = 'No faces'
            else:
                if score != 0:
                    text = 'BlackList'
                    self.play = False
                    color = [1, 0.1, 0.1, 1]
                else:
                    text = 'Access approved'
                    color = [0.1, 0.1, 1, 1]

            label.text = text
            # detections = detect_faces(frame)
            # img = Image.fromarray(frame)
            # draw = ImageDraw.Draw(img)
            # for det in detections:
            #     x1, y1, x2, y2 = det.bbox.as_tuple

            #     shape = [int(self.texture.width*x1), int(self.texture.height*y1),
            #             int(self.texture.width*x2), int(self.texture.height*y2)]
            #     draw.rectangle(xy=shape)
            # frame = np.array(img)
        except Exception as e:
            if platform != 'android':
                raise e
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            tb = '\n'.join(tb)
            tb = [tb[i:i + 50] for i in range(0, len(tb), 50)]
            tb = '\n'.join(tb)
            self.parent.children[0].text = tb[-100:]

        return frame

    def put_frame(self, frame):
        buf = frame.tobytes()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')


class CameraClick(Screen):
    def on_texture(self, instance):
        print(self)

    # def capture(self):
    #     '''
    #     Function to capture the images and give them the names
    #     according to their captured time and date.
    #     '''
    #     camera = self.ids['camera']
    #     timestr = time.strftime("%Y%m%d_%H%M%S")
    #     newvalue = np.frombuffer(camera.texture.pixels, np.uint8)
    #     print(newvalue.shape)
    #     height, width = camera.texture.height, camera.texture.width

    #     newvalue = newvalue.reshape(height, width, 4)

    #     print(newvalue.shape)
    #     camera.export_to_png("IMG_{}.png".format(timestr))
    #     print("Captured")


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class UpdateWindow(Screen):
    home_folder = os.getcwd()+'/BlackListImages'
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def get_default_path(self):
        self.path = self.home_folder
        return self.path  # I added the return statement

    def load(self, path, filename):
        print(self, path, filename)
        print(self.home_folder)

        shutil.copy(filename[0], self.home_folder)
        # with open(os.path.join(path, filename[0])) as stream:
        #     self.text_input.text = stream.read()

        self.dismiss_popup()
        self.ids.filechooser._update_files()

    def selected(self, filename):
        self.ids.image.source = filename[0]

    def sync_press(self):
        print('h')
        face_data.update_database()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_only_images(self, directory, filename):
        return filename.endswith(".jpg") or \
            filename.endswith(".png") or \
            filename.endswith(".jpeg")


class WindowManager(ScreenManager):
    pass


class MyApp(App):
    def build(setLevel):
        check_request_camera_permission()
        kv = Builder.load_file('check.kv')
        return kv


if __name__ == '__main__':
    MyApp().run()
