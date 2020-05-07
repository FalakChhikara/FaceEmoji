import cv2
from django.shortcuts import render, redirect
import numpy as np
import pathlib
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

filepath='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 1

    y_pred = K.clip(y_pred, 0, 1)

    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))




def homepage(request):
    return render(request, 'facetoemoji/home.html')

def _locate_faces(image):
    faces = faceCascade.detectMultiScale(
        image
    )
    return faces  # list of (x, y, w, h)

def find_faces(image):
    faces_coordinates = _locate_faces(image)
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]
    normalized_faces = [_normalize_face(face) for face in cutted_faces]
    return zip(normalized_faces, faces_coordinates)

def _normalize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (350, 350))
    return face




def expr(image,model):
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image,(48,48))
    image = np.stack((image,)*1, axis=-1)
    image = np.expand_dims(image, axis=0)
    arr = model.predict(image)
    # print(arr)
    result = arr[0].argmax()
    return result






def webcam(request):
    cap = None

    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cdir = str(pathlib.Path(__file__).parent.absolute())
    modelpath = cdir + '\\' + 'weights.h5'

    model = keras.models.load_model(modelpath, custom_objects={"fbeta": fbeta})
    # emotions = ['anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    while True:
        check, frame = video.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Face', frame)
        print(type(frame))
        print(modelpath)
        for face, (x, y, w, h) in find_faces(frame):

            prediction = expr(face, model)
            # /content/4.png
            idir = cdir + '\\' + 'graphics' + '\\' + str(prediction) + '.png'
            print(idir)

            em = cv2.imread(idir)
            print(type(em))
            # em = cv2.cvtColor(em, cv2.COLOR_RGB2BGR)
            em = cv2.resize(em, (w, h))
            frame[y:y + h, x:x + w] = em


            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 50)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2

            cv2.putText(frame, 'Press Q to quit',
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            cv2.imshow('emoji', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    return redirect('facetoemoji:homepage')
