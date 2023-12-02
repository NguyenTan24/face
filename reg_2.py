import cv2
import os
import numpy as np

# Tạo thư mục chính để lưu trữ ảnh của từng người
base_folder = 'face_data'
face_cascade_folder = cv2.data.haarcascades
recognizer_model = 'recognizer_model.yml'

# Kiểm tra xem mô hình recognizer đã được tạo chưa, nếu chưa tạo, hãy tạo mới
if not os.path.exists(recognizer_model):
    print("Mô hình recognizer chưa được tạo. Hãy tạo mô hình trước.")
    exit()

# Tạo bộ phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier(os.path.join(face_cascade_folder, 'haarcascade_frontalface_default.xml'))

# Tạo đối tượng recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(recognizer_model)

# Mở webcam (0 là camera mặc định, nếu có nhiều camera, hãy thay đổi giá trị này)
cap = cv2.VideoCapture(0)

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    # Chuyển ảnh sang đa cường độ màu để xử lý nhanh hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sử dụng bộ phân loại để nhận diện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Nhận diện khuôn mặt và trả về id và độ chính xác
        label, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Nếu độ chính xác tốt, hiển thị tên của người đó
        if confidence < 100:
            person_folder = os.path.join(base_folder, f'person_{label}')
            person_name = f'Person {label}'
            cv2.putText(frame, f"{person_name} ({confidence:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị khung hình kết quả
    cv2.imshow('Face Recognition', frame)

    # Đợi 1 giây để người dùng có thể nhìn thấy mỗi frame
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
