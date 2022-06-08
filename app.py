from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

from DetectOrange import DetectOrange

Detector = DetectOrange()

@app.route('/')
def index():
    return render_template('index.html')


def gen():
    cap = cv2.VideoCapture('./Sources/Video/3.mp4')

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture image")
            break

        bboxs = Detector.inference(frame)
        if len(bboxs) > 0:
            for bb in bboxs:
                Detector.draw(frame, bb)

        cv2.imwrite('demo.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)