import cv2
import os

# Tạo bộ phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Thư mục chính lưu trữ kho khuôn mặt
base_folder = 'path/to/your/face_data/'

# Duyệt qua từng thư mục trong thư mục chính
for person_folder in os.listdir(base_folder):
    person_folder_path = os.path.join(base_folder, person_folder)

    # Nếu là thư mục, tiến hành nhận diện khuôn mặt
    if os.path.isdir(person_folder_path):
        # Duyệt qua từng file ảnh trong thư mục người đó
        for filename in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, filename)

            # Đọc ảnh từ đường dẫn
            image = cv2.imread(image_path)

            # Chuyển đổi ảnh sang đa cường độ màu
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Sử dụng bộ phân loại để nhận diện khuôn mặt
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            # Duyệt qua từng khuôn mặt và lưu vào thư mục đầu ra
            for i, (x, y, w, h) in enumerate(faces):
                # Cắt khuôn mặt từ ảnh gốc
                face_image = image[y:y+h, x:x+w]

                # Lưu khuôn mặt vào thư mục đầu ra với tên định dạng: person_folder/face_<original_filename>_<index>.png
                output_path = os.path.join(person_folder_path, f"face_{filename}_{i+1}.png")
                cv2.imwrite(output_path, face_image)

print("Đã lưu khuôn mặt vào kho lưu trữ.")
