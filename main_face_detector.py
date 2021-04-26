from kivy.app import App
from kivy.lang import Builder
from kivy.clock import mainthread
from kivy.logger import Logger
from kivy.utils import platform
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
import kivy
import numpy as np
from face_detection import detect_faces
import traceback
import sys




def is_android():
    return platform == 'android'

'''
Runtime permissions:
'''

def check_camera_permission():
    """
    Android runtime `CAMERA` permission check.
    """
    if not is_android():
        return True
    from android.permissions import Permission, check_permission
    permission = Permission.CAMERA
    return check_permission(permission)


def check_request_camera_permission(callback=None):
    """
    Android runtime `CAMERA` permission check & request.
    """
    had_permission = check_camera_permission()
    Logger.error("CameraAndroid: CAMERA permission {%s}.", had_permission)
    if not had_permission:
        Logger.info("CameraAndroid: CAMERA permission was denied.")
        Logger.info("CameraAndroid: Requesting CAMERA permission.")
        from android.permissions import Permission, request_permissions
        permissions = [Permission.CAMERA]
        request_permissions(permissions, callback)
        had_permission = check_camera_permission()
        Logger.info("CameraAndroid: Returned CAMERA permission {%s}.", had_permission)
    else:
        Logger.info("CameraAndroid: Camera permission granted.")
    return had_permission

class MyCamera(Camera):
    def __init__(self, **kwargs):
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
        frame = np.frombuffer(self._camera.texture.pixels, np.uint8).reshape(h, w, 4)[..., :3]
        
        frame = self.process_frame(frame)
        
        self.put_frame(frame)
        super(MyCamera, self).on_tex(*l)

    def process_frame(self, frame):
        try:
            detections = detect_faces(frame)
            self.parent.children[0].text = str(len(detections))
            
            
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            tb = '\n'.join(tb)
            tb = [tb[i:i + 50] for i in range(0, len(tb), 50)]
            tb = '\n'.join(tb)
            self.parent.children[0].text = tb
        
        return frame
    
    def put_frame(self, frame):
        buf = frame.tobytes()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

camUI = '''
BoxLayout:
    size: root.size
    # size_hint: 1,1
    canvas:
        Color:
            rgb: 1, 0, 0
        Rectangle:
            size: self.size
        Rotate:
            angle: -90
            origin: self.center
    MyCamera:
        id: cam0
        resolution: 1920, 1080
        play: True
        # size_hint: 2, 2
        allow_stretch: True
    
    Label:
        text: "Hello"
        
'''


class TestCamera(App):

    def build(self):
        
        check_request_camera_permission()
        return Builder.load_string(camUI)
        # else:
            # return Builder.load_string(perm_denied)


TestCamera().run()
