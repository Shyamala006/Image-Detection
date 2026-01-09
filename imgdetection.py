import cv2
import numpy as np
import matplotlib.pyplot as plt

# COMMON FACE DETECTOR
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# IMAGE PROCESSING FUNCTION
def process_image(image):
    image = cv2.resize(image, (500, 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.equalizeHist(gray)
    edges = cv2.Canny(blur, 50, 150)

    # Object detection
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    object_image = image.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(object_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Face detection
    face_image = image.copy()
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(face_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image, gray, enhanced, edges, object_image, face_image

# MENU
print("IMAGE PROCESSING USING OPENCV")
print("1. Process Image File")
print("2. Real-Time Webcam Processing")

choice = input("Enter your choice (1 or 2): ")

# OPTION 1: IMAGE FILE INPUT
if choice == "1":
    image_path = input("Enter image path: ")
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found!")
        exit()

    results = process_image(image)

    titles = [
        "Original Image",
        "Grayscale",
        "Enhanced",
        "Edge Detection",
        "Object Detection",
        "Face Detection"
    ]

    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        if len(results[i].shape) == 2:
            plt.imshow(results[i], cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# OPTION 2: WEBCAM INPUT
elif choice == "2":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not accessible")
        exit()

    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image, gray, enhanced, edges, object_img, face_img = process_image(frame)

        cv2.imshow("Real-Time Processing", face_img)
        cv2.imshow("Edge Detection", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid choice! Please select 1 or 2.")