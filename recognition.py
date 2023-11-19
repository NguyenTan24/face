import cv2
import os

# Đường dẫn đến thư mục chứa ảnh
image_folder = 'path/to/your/images/'

# Tạo bộ phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Thư mục để lưu ảnh đã được nhận diện
output_folder = 'path/to/output/images/'
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua tất cả các file ảnh trong thư mục
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Đường dẫn đầy đủ đến file ảnh
        image_path = os.path.join(image_folder, filename)

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

            # Lưu khuôn mặt vào thư mục đầu ra với tên định dạng: output_folder/face_<original_filename>_<index>.png
            output_path = os.path.join(output_folder, f"face_{filename}_{i+1}.png")
            cv2.imwrite(output_path, face_image)

# Hiển thị số lượng khuôn mặt đã được nhận diện
print(f"Đã nhận diện khuôn mặt từ {len(os.listdir(output_folder))} ảnh.")