import os
# from keras_facenet import FaceNet
import numpy as np
from model import TensorFlowModel
from face_detection import detect_faces
from PIL import Image

def is_face_vertical(data):
    right_eye = data[3]
    left_eye = data[2]
    angle = np.arctan2(*((right_eye - left_eye)[::-1])) / np.pi  * 180
    return abs(angle) < 15

class Embedder:

    def __init__(self, folder):
        self.threshold = 0.7
        self.model = TensorFlowModel()
        self.model.load(os.path.join(os.getcwd(), 'face_detection/data/face_embedding.tflite'), num_threads=2)
        # self.model.resize_input((1, 80, 80, 3))
        self.home_folder = folder
        self.db_folder = 'db'
        if not os.path.exists(self.db_folder):
            os.mkdir(self.db_folder)
        self.db = {}
        self.db_names = []
        self.update_database(folder)

    def preprocessing(self, img: Image):
        image_size = 160, 160
        # image_size = 80, 80
        
        img = img.resize(image_size, resample=Image.BILINEAR)
        img = np.expand_dims(np.array(img)[..., :3], axis=0)
        return (np.float32(img) - 127.5) / 127.5

    def get_embedding_on_face(self, face_img: Image):
        if isinstance(face_img, np.ndarray):
            face_img = Image.fromarray(face_img, mode='RGB')
        face_img = self.preprocessing(face_img)
        emb = self.model.pred(face_img)
        emb = emb/np.linalg.norm(emb)
        return emb

    def is_image(self, filename):
        return filename.endswith(".jpg") or \
            filename.endswith(".png") or \
            filename.endswith(".jpeg")

    def find_face(self, face_img):
        if self.db:
            y = self.get_embedding_on_face(face_img)
            x = np.vstack(list(self.db.values()))

            scores = (x*y).sum(axis=1)
            max_idx = scores.argmax()
            print(scores)
            if scores[max_idx] > self.threshold:
                return list(self.db.keys())[max_idx], scores[max_idx]
        return None, 0.
    
    def find_person(self, img):
        # img is bgr picture
        detections = detect_faces(img)
        if detections:
            det = max(detections, key=lambda x: x.bbox.area)
            box = det.bbox.scale(img.shape[1::-1]).as_tuple
            
            if not is_face_vertical(det.data):
                return False, (None, 0.)

            x1, y1, x2, y2 = [int(i) for i in box]
            face = img[y1:y2, x1:x2]
            if face.size > 0:
                return True, self.find_face(face)
        return False, (None, 0.)
        

    def update_database(self, directory):
        try:
            images = set(filename for filename in os.listdir(directory)
                                                if self.is_image(filename))
        except FileNotFoundError:
            images = set()
        npys = set(npy for npy in os.listdir(self.db_folder) if npy.endswith('.npy'))
        
        for filename in images:
            emb_file = filename.split('.')[0]+'.npy'
            if emb_file not in npys:
                path = os.path.join(self.home_folder, filename)
                img = Image.open(path)
                img = np.asarray(img)
                detections = detect_faces(img)
                if detections:
                    print("Add", filename, 'to database')
                    det = max(detections, key=lambda x: x.bbox.area)
                    if not is_face_vertical(det.data):
                        os.remove(os.path.join(directory, filename))
                        continue

                    box = det.bbox.scale(img.shape[1::-1]).as_tuple
                    x1, y1, x2, y2 = [int(i) for i in box]
                    face = img[y1:y2, x1:x2]
                    emb = self.get_embedding_on_face(face)
                    np.save(os.path.join(self.db_folder, emb_file), emb)
                else:
                    os.remove(os.path.join(directory, filename))
                    continue

            if filename not in self.db:
                self.db[emb_file] = np.load(os.path.join(self.db_folder, emb_file))

        if images:
            images = set(filename.split('.')[0] for filename in os.listdir(directory) 
                                                    if self.is_image(filename))
        for npy in npys:
            if npy.split('.')[0] not in images:
                print("Remove", npy, 'from database')
                os.remove(os.path.join(self.db_folder, npy))
                if npy in self.db:
                    self.db.pop(npy)


if __name__ == '__main__':
    inst = Embedder('BlackListImages')
    # inst.update_database()
    img = cv2.imread('jj.png')
    print(inst.find_person(img))
