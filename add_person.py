import os
import cv2

from settings import CAMERA_SOURCE, USER_IMAGES_FOLDER, IMAGE_AMOUNT_FOR_TRAIN

def capture_and_save(full_name):
    joined_full_name = '_'.join(full_name.split(' '))
    cv2.namedWindow('Capture and Save User Photo')
    capture = cv2.VideoCapture(CAMERA_SOURCE)
    face_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}/haarcascade_frontalface_alt2.xml')

    img_path = os.path.join(USER_IMAGES_FOLDER, joined_full_name)
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    counter = 0
    while capture.isOpened():
        flag, frame = capture.read()
        if not flag:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print("========>", grey)
        face_rect = face_cascade.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

        if len(face_rect) > 0:
            for face in face_rect:
                x, y, w, h = face

                img_name = f'{joined_full_name}_{counter}.png'
                cropped_img = frame[y - 10: y + h + 10, x - 5: x + w + 5]
                cv2.imwrite(f'{img_path}/{img_name}', cropped_img)

                cv2.rectangle(frame, (x - 5, y - 10), (x + w + 5, y + h + 10), (0, 0, 255), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f'{full_name} {counter}', (x + 30, y - 15), font, 1, (0, 250, 250), 4)

                counter += 1
                if counter > IMAGE_AMOUNT_FOR_TRAIN:
                    break

            if counter > IMAGE_AMOUNT_FOR_TRAIN:
                break

        cv2.imshow('Capture and Save User Photo', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    full_name = input('Enter user full name: ')
    capture_and_save(full_name)