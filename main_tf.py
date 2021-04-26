import os
import kivy
import numpy as np
from kivy.app import App
from kivy.uix.label import Label
from kivy.utils import platform
import traceback
import sys

class MyApp(App):

    def build(self):
        try:
            if platform == 'android':
                from face_detection.android import FaceDetection
                detect_faces = FaceDetection(num_threads=3)
            else:
                from face_detection.desktop import FaceDetection
                detect_faces = FaceDetection()
            y = detect_faces("./arkady.jpg")
            
            print(y[0].bbox)
            
            # result should be
            # 0.01647118,  1.0278152 , -0.7065112 , -1.0278157 ,  0.12216613,
            # 0.37980393,  0.5839217 , -0.04283606, -0.04240461, -0.58534086
            return Label(text=f'{y[0].bbox}')
        except Exception as e:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb = traceback.format_exception(exc_type, exc_value, exc_tb)
            tb = '\n'.join(tb)
            tb = [tb[i:i + 50] for i in range(0, len(tb), 50)]
            tb = '\n'.join(tb)
            e = str(e)
            e = [e[i:i + 50] for i in range(0, len(e), 50)]
            e = '\n'.join(e)
            return Label(text=f"{e}\n{tb}"[-1000:])


if __name__ == '__main__':
    MyApp().run()
