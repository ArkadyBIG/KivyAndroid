from kivy.utils import platform

if platform == 'android':
    from face_detection.android import FaceDetection
    detect_faces = FaceDetection(num_threads=3)
else:
    from face_detection.desktop import FaceDetection
    detect_faces = FaceDetection()

