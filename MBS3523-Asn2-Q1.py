import cv2
import face_recognition

# Load images and encode known faces
person1_image = face_recognition.load_image_file("person1.jpg")
person1_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file("person2.jpg")
person2_encoding = face_recognition.face_encodings(person2_image)[0]

person3_image = face_recognition.load_image_file("person3.jpg")
person3_encoding = face_recognition.face_encodings(person3_image)[0]

known_encodings = [person1_encoding, person2_encoding, person3_encoding]
known_names = ["Person 1", "Person 2", "Person 3"]

# Load and encode your own image
my_image = face_recognition.load_image_file("my_face.jpg")
my_encoding = face_recognition.face_encodings(my_image)[0]

# Set up video capture device
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for any known face
        matches = face_recognition.compare_faces(known_encodings + [my_encoding], face_encoding)

        # If there's a match, display the name and score on the top left corner of the bounding box
        # If there's no match, display "Unknown Person"
        name = "Unknown Person"
        score = None
        for i, match in enumerate(matches):
            if match:
                if i == len(matches) - 1:
                    name = "Oscar Sin"
                    score = 220669394
                else:
                    name = known_names[i]
                    face_distances = face_recognition.face_distance([known_encodings[i]], face_encoding)
                    score = round((1 - face_distances[0]) * 100, 2)
                break

        # Draw a box around the face and display the name and score
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, f"{name} {score}%", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
