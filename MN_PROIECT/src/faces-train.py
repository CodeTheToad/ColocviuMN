# Folosim aceasta librarie pentru a
# Naviga prin fisiere si a gasi
# Fisierele corespunzatoare
import os
import cv2
import numpy as np
# ============================
# 
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR", BASE_DIR)
image_dir = os.path.join(BASE_DIR, "images")
print("image_dir", image_dir)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	#print("root", root)
	for file in files:
		#print("file", file)
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			#print(label, path)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]

			print(label_ids)			
			# Convertim imagiena la grayscale
			pil_image = Image.open(path).convert("L")
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			#print(image_array)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)


print(y_labels)
print(x_train)

with open("pickles/face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/trainner.yml")