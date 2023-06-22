import numpy as np
import cv2
import os
import PySimpleGUI as sg

FREQ_DIV = 5
RESIZE_FACTOR = 4
NUM_TRAINING = 100

class TrainFaces:
    def __init__(self, name):
        cascPath = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        self.face_dir = 'face_data'
        self.face_name = name
        self.path = os.path.join(self.face_dir, self.face_name)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.count_captures = 0
        self.count_timer = 0
        self.capture = cv2.VideoCapture(0)

    def capture_training_images(self):
        while True:
            self.count_timer += 1
            ret, frame = self.capture.read()
            inImg = np.array(frame)
            outImg = self.process_image(inImg)
            cv2.imshow('Video', outImg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.capture.release()
                cv2.destroyAllWindows()
                return

    def process_image(self, inImg):
        frame = cv2.flip(inImg, 1)
        resized_width, resized_height = (112, 92)
        if self.count_captures < NUM_TRAINING:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (int(gray.shape[1] / RESIZE_FACTOR), int(gray.shape[0] / RESIZE_FACTOR)))
            faces = self.face_cascade.detectMultiScale(
                gray_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                areas = []
                for (x, y, w, h) in faces:
                    areas.append(w * h)
                max_area, idx = max([(val, idx) for idx, val in enumerate(areas)])
                face_sel = faces[idx]

                x = face_sel[0] * RESIZE_FACTOR
                y = face_sel[1] * RESIZE_FACTOR
                w = face_sel[2] * RESIZE_FACTOR
                h = face_sel[3] * RESIZE_FACTOR

                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (resized_width, resized_height))
                img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(self.path) if fn[0] != '.'] + [0])[-1] + 1

                if self.count_timer % FREQ_DIV == 0:
                    cv2.imwrite('%s/%s.png' % (self.path, img_no), face_resized)
                    self.count_captures += 1
                    print("Captured image: ", self.count_captures)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, self.face_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        elif self.count_captures == NUM_TRAINING:
            sg.popup_auto_close("Press to 'q' to train")
            self.count_captures += 1

        return frame

    def train_data(self):
        imgs = []
        tags = []
        index = 0

        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                img_path = os.path.join(self.face_dir, subdir)
                for fn in os.listdir(img_path):
                    path = img_path + '/' + fn
                    tag = index
                    imgs.append(cv2.imread(path, 0))
                    tags.append(int(tag))
                index += 1
        (imgs, tags) = [np.array(item) for item in [imgs, tags]]

        self.model.train(imgs, tags)
        self.model.save('trained_data.xml')
        print("Training completed successfully")
        return

class RecogFaces:
    def __init__(self):
        cascPath = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        self.face_dir = 'face_data'
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.face_names = []
        self.capture = cv2.VideoCapture(0)

    def load_trained_data(self):
        names = {}
        key = 0
        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                names[key] = subdir
                key += 1
        self.names = names
        self.model.read('trained_data.xml')

    def show_video(self):
        sg.popup_auto_close("Press to 'q' to exit")
        while True:
            ret, frame = self.capture.read()
            inImg = np.array(frame)
            outImg, self.face_names = self.process_image(inImg)
            cv2.imshow('Video', outImg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.capture.release()
                cv2.destroyAllWindows()
                return

    def process_image(self, inImg):
        frame = cv2.flip(inImg,1)
        resized_width, resized_height = (112, 92)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (int(gray.shape[1] / RESIZE_FACTOR), int(gray.shape[0] / RESIZE_FACTOR)))
        faces = self.face_cascade.detectMultiScale(
                gray_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
        persons = []
        for i in range(len(faces)):
            face_i = faces[i]
            x = face_i[0] * RESIZE_FACTOR
            y = face_i[1] * RESIZE_FACTOR
            w = face_i[2] * RESIZE_FACTOR
            h = face_i[3] * RESIZE_FACTOR
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (resized_width, resized_height))
            confidence = self.model.predict(face_resized)
            if confidence[1]<100:
                person = self.names[confidence[0]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 3)
                cv2.putText(frame, '%s - %.0f' % (person, confidence[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))
            else:
                person = 'Unknown'
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame, person, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))
            persons.append(person)
        return (frame, persons)

def run():
    layout = [[sg.Text('Enter New Name'), sg.InputText(key='-IN-')],
              [sg.Button('Train'), sg.Button('Test')]]

    window = sg.Window('Face recognition', layout)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Train':
            name = values['-IN-']
            trainer = TrainFaces(name)
            trainer.capture_training_images()
            trainer.train_data()
            sg.popup_auto_close('Done')
        elif event == 'Test':
            recognizer = RecogFaces()
            recognizer.load_trained_data()
            recognizer.show_video()
    window.close()

if __name__ == '__main__':
    run()
