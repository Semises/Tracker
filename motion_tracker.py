import cv2

# Importing video to a variable
captured_video = cv2.VideoCapture("video.avi")

# Taking two frames one after another
_, first_frame = captured_video.read()
_, second_frame = captured_video.read()

while captured_video.isOpened():

    # Movement as a diffrence between two frames
    diff = cv2.absdiff(first_frame, second_frame)

    # Changing from BGR to Grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Eliminating noise
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # Setting value of pixels according to threshold
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilating for thicker contours
    dilation = cv2.dilate(thresh, None, 3)

    # Finding contours and drawing rectangles for movement
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 850:
            continue
        cv2.rectangle(first_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    
    # Showing video frame-by-frame
    cv2.imshow("Video", first_frame)
    first_frame = second_frame
    _, second_frame = captured_video.read()

    # Waiting for ESC for 60ms
    key = cv2.waitKey(60)
    if key == 27:
        break 

# Releasing the resources
captured_video.release()
cv2.destroyAllWindows()