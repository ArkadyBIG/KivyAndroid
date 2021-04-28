import kivy

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.label import Label
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
from threading import Thread
from kivy.clock import Clock
if platform != 'android':
    import matplotlib.pyplot as plt
    
    

if not os.path.exists(os.getcwd() + '/BlackListImages'):
    os.mkdir(os.getcwd() + '/BlackListImages')

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
        return False
    return True


class MainWindow(Screen):
    pass

def crop_content(frame, area):
    
    x, y = area
    
    dy = (frame.shape[0] - x) // 2
    
    return frame[dy:-dy]


face_data = Embedder(os.getcwd() + '/BlackListImages')
class MyCamera(Camera):
    def __init__(self, **kwargs):
        if kivy.platform == 'android':
            self.resolution = 1920, 1080
        #self.face_data = Embedder(os.getcwd() + '/BlackListImages')\
        self._skip_frames = 0
        self.thread = None
        
        super(MyCamera, self).__init__(**kwargs)
        Clock.schedule_once(self.get_frame, 3)

    def get_frame(self, delta):
        
        if (not self.thread or not self.thread.is_alive()) \
                        and self.play and self._camera.texture:
            if self._skip_frames > 0:
                self._skip_frames -= 1
                self.parent.ids['label'].text='skipping'
                return Clock.schedule_once(self.get_frame, 0.2)
                
            label = self.parent.ids['label']
            try:
                frame = np.frombuffer(self._camera.texture.pixels, 'uint8')
                self.thread = Thread(target=self.process_frame, args=(frame,))
                self.thread.start()
            except Exception as e:
                if platform != 'android':
                    raise e
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb = traceback.format_exception(exc_type, exc_value, exc_tb)
                tb = '\n'.join(tb)
                tb = [tb[i:i + 50] for i in range(0, len(tb), 50)]
                tb = '\n'.join(tb)
                label.text = tb[-100:]
        Clock.schedule_once(self.get_frame, 0.2)
        

    def process_frame(self, frame):
        try:
            # # emb =
            # print(self.parent.ids['label'])
            # print(self.parent.children[0])
            # print(self.parent.children[1])
            #label = self.ids['label']#self.parent.children[0]#.children[1]
            w, h = self._camera._resolution
            frame = frame.reshape(h, w, 4)
            frame = frame[..., :3]
            label = self.parent.ids['label']
            
            if platform != 'android':
                frame = frame
            else:
                frame = np.rot90(frame, 3)
            face_found, (name, score) = data = face_data.find_person(frame)
            if not face_found:
                color = [0.1, 0.1, 0.1, 1]
                text = 'No faces'
            else:
                if score != 0:
                    text = 'BlackList'
                    self.play = False
                    self._skip_frames = 6
                    color = [1, 0.1, 0.1, 1]
                else:
                    text = 'Access approved'
                    color = [0.1, 0.1, 1, 1]
            label.text = text
            # labe = text
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
            label.text = tb[-100:]

        return frame

class CameraClick(Screen):
    def on_texture(self, instance):
        print(self)

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class UpdateWindow(Screen):
    home_folder = os.getcwd() + '/BlackListImages'
    
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
        face_data.update_database(self.ids.filechooser.path)

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        content.ids.filechooser.rootpath = '/storage/emulated/0' if platform == 'android' else '/home'
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
        try:
            if check_request_camera_permission():
                
                kv = Builder.load_file('check.kv')
                return kv
            else:
                return Label(text='reload')
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            tb = '\n'.join(tb)
            tb = [tb[i:i + 50] for i in range(0, len(tb), 50)]
            tb = '\n'.join(tb)
            text = tb[-500:]
            return Label(text=text)
            


if __name__ == '__main__':
    MyApp().run()
