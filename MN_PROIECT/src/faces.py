# Reprezentarea imaginilor
# Sub o forma matriciala cu valori
import numpy as np

# ===============================
# Procesare imagine, clasificator
# Si creeare trainer
import cv2

# ===============================
# Folosit la recunoasterea emotiei
#  [CNN] > Convolution Neural Network
from deepface import DeepFace

# ===============================
# Folosit pentru a stoca valorile
# Imagnilor intr-un fisier binar
import pickle



# =================================================
# Folosim si initializam clasificatorul
# Pus la dispozitie de libraria openCV
face_cascade =cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/trainner.yml")

#=============================================
# Deschidem fisierul binar cu valorile
# Fiecarei imagini stocate sub forma de matrici
with open("pickles/face-labels.pickle", 'rb') as f:
    label_ids = pickle.load(f)
    labels = {v: k for k, v in label_ids.items()}

# Deschidem camera web sau videoclipul
cap = cv2.VideoCapture('videoTest2.mp4')

while (True):
    # Capturam cadru-cu cadru imaginea provenita
    # De la camera web sau videoclip
    ret, frame = cap.read()
    # Convertim frame-urile colorate la grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # print(x,y,w,h) # Aici putem printa coordonatele fetei in fiecare frame
        roi_gray = gray[y:y+h, x:x+w]  # (ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]

        # ==============================================
        # Mai jos folosim recognizer-ul pus la dipozitie
        # De libraria openCV, folosindu-ne de 
        # Trainner-ul creeat de noi
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 4 and conf <= 85:
            # =======================
            # Afisam numele persoanei 
            # Identificate in consola
            print(labels[id_]) 
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            
            # =================================
            # Preluam rezultatul de la DeepFace.analyze
            # Si il salvam in 'predictie'
            predictie = DeepFace.analyze(frame)
            # Salvam in 'emotie' doar ceea ce ne intereseaza
            # Si anume 'dominant_emotion' din analiza efectuata
            emotie = predictie['dominant_emotion']

            # ===============================
            # Pozitie pentru afisarea emotiei
            cv2.putText(frame, emotie, (x, y+200), font, 1, color, stroke, cv2.LINE_AA) 
            # Pozitie afisare nume persoana
            cv2.putText(frame, name, (x, y-10), font, 1, color, stroke, cv2.LINE_AA) 

        # ====================================
        # Mai jos avem codul pentru desenarea 
        # conturului in jurul fetei
        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # ===============================================
    # Afisam imaginea preluata de la webcam/videoclip
    cv2.imshow('VIDEO/WEBCAM', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# ==========================================================
# Dupa ce apasam tasta 'q', fereastra cu video-ul se inchide 
cap.release()
cv2.destroyAllWindows()
