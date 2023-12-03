import cv2
import os

# Tạo thư mục chính để lưu trữ khuôn mặt
base_folder = 'dataSet'
os.makedirs(base_folder, exist_ok=True)

# Tạo đối tượng để truy cập webcam (0 là camera mặc định, nếu có nhiều camera, hãy thay đổi giá trị này)
cap = cv2.VideoCapture(0)

# Tạo bộ phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Biến đếm số frame đã chụp
frame_count = 0


folder_name = input('Nhập tên mới: ')
person_folder = os.path.join(base_folder,folder_name)
os.makedirs(person_folder, exist_ok=True)
print('Folder created')

while frame_count < 10:

    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    # Chuyển ảnh sang đa cường độ màu để xử lý nhanh hơn
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sử dụng bộ phân loại để nhận diện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Tạo thư mục để lưu khuôn mặt, tên thư mục dựa trên chỉ số của khuôn mặt
        person_folder = os.path.join(base_folder, f'person_{frame_count + 1}')
        os.makedirs(person_folder, exist_ok=True)

        # Lưu hình ảnh khuôn mặt
        face_image = frame[y:y + h, x:x + w]
        face_image_path = os.path.join(person_folder, f"face_{frame_count + 1}.png")
        cv2.imwrite(face_image_path, face_image)
        frame_count += 1

    # Hiển thị khung hình kết quả
    cv2.imshow('Capture Faces', frame)

    # Đợi 1 giây để người dùng có thể nhìn thấy mỗi frame
    cv2.waitKey(1000)
    print('Frame saved!')
# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
