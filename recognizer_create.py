import cv2
import os
import numpy as np

# Tạo thư mục chính để lưu trữ ảnh của từng người
base_folder = 'face_data'

# Tạo bộ phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tạo đối tượng recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Danh sách để lưu trữ ảnh và nhãn tương ứng
faces = []
labels = []

# Duyệt qua các thư mục con (người) trong thư mục chính
for person_folder in os.listdir(base_folder):
    person_path = os.path.join(base_folder, person_folder)
    
    # Đọc nhãn từ tên thư mục
    label = int(person_folder.split('_')[1])
    
    # Duyệt qua các file ảnh trong thư mục người đó
    for filename in os.listdir(person_path):
        image_path = os.path.join(person_path, filename)
        
        # Đọc ảnh và chuyển đổi sang đa cường độ màu
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Nhận diện khuôn mặt trong ảnh
        faces_rect = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces_rect:
            # Lưu khuôn mặt và nhãn tương ứng vào danh sách
            faces.append(img[y:y+h, x:x+w])
            labels.append(label)

# Train recognizer với danh sách ảnh và nhãn
recognizer.train(faces, np.array(labels))

# Lưu mô hình vào file
recognizer.save('recognizer_model.yml')
