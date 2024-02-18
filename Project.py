import os
from flask import Flask, request, render_template, Response
import cv2
import dlib

app = Flask(__name__, template_folder='templates')

def generate_frames(video_path, output_folder):
    # Open the video file for capturing frames
    video_capture = cv2.VideoCapture(video_path)

    # Counter for frames and set to store detected faces to avoid duplicates
    frame_count = 0
    detected_faces = set()

    # Load the pre-trained face detector from dlib
    detector = dlib.get_frontal_face_detector()

    while True:
        # Read the next frame from the video
        ret, frame = video_capture.read()
        if not ret:
            # Break the loop if the video ends
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection using dlib
        faces = detector(gray_frame)

        for face_num, face in enumerate(faces):
            # Get the coordinates of the detected face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            # Check if the face has already been detected
            face_id = f"{x}_{y}_{w}_{h}"
            if face_id not in detected_faces:
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop and save the detected face
                face_image = frame[y:y + h, x:x + w]
                save_path = os.path.join(output_folder, f"face_{frame_count}_{face_num}.jpg")
                cv2.imwrite(save_path, face_image)

                # Mark the face as detected to avoid duplicates
                detected_faces.add(face_id)

        frame_count += 1

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the video capture object
    video_capture.release()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('upload.html', message='No file part')

        file = request.files['file']

        # Check if the file is selected
        if file.filename == '':
            return render_template('upload.html', message='Please Select the video file🙏🏻')

        if file:
            # Automatically create the "uploads" folder if it doesn't exist
            uploads_folder = os.path.join(app.root_path, 'uploads')
            os.makedirs(uploads_folder, exist_ok=True)

            # Save the uploaded video file to the "uploads" folder
            video_path = os.path.join(uploads_folder, file.filename)
            file.save(video_path)
            output_folder = "output_faces"

            # Create the output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Use Response to return the video stream
            return Response(generate_frames(video_path, output_folder),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

    return render_template('upload.html', message=' ')


if __name__ == "__main__":
    # Run the Flask app in debug mode
    app.run(debug=True)
