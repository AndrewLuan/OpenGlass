import cv2  
  
cap = cv2.VideoCapture(0)  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
  
while True:  
    ret, frame = cap.read()  
    if ret:  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  
        for (x, y, w, h) in faces_rects:  
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
        cv2.imshow('Face Detection', frame)  
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  
    else:  
        break

cap.release()  
cv2.destroyAllWindows()