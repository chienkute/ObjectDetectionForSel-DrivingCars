from flask import Flask, render_template, Response,send_file,request,session,make_response
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
import requests
import time
import base64
import numpy as np
from ultralytics import YOLO
# Required to run the YOLOv8 model
import cv2
model = YOLO("E:/ĐATN/Livestream/YOLOv8-CrashCourse/Weights/best.pt")
# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import video_detection
app = Flask(__name__)

app.config['SECRET_KEY'] = 'muhammadmoin'
app.config['UPLOAD_FOLDER'] = 'static/files'
#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")
# def generate_frames(path_x = ''):
#     yolo_output = video_detection(path_x)
#     for detection_,color  in yolo_output:
#         ref,buffer=cv2.imencode('.jpg',detection_)
#         frame=buffer.tobytes()
#         yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n' + color.encode() + b'\r\n')
def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('indexproject.html')
@app.route("/webcam", methods=['GET','POST'])
def webcam():
    session.clear()
    return render_template('ui.html')
@app.route('/FrontPage', methods=['GET','POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoprojectnew.html', form=form)
@app.route('/video')
def video():
    # return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(video_detection(session.get('video_path', None)),mimetype='text/event-stream')


# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    #return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect_objects():
    data = request.get_json()
    image_data = data['imageData']  # Giả sử dữ liệu ảnh base64 được gửi trong trường 'imageData'

    # Giải mã chuỗi base64 thành dữ liệu ảnh
    # image_data = image_data.split(',')[1]  # Bỏ qua tiền tố 'data:image/jpeg;base64,'
    # image_bytes = base64.b64decode(image_data)
    # image_array = np.frombuffer(image_bytes, np.uint8)
    # image = cv2.imdecode(image_array, cv2.IMREA   D_COLOR)
    response = requests.get(image_data)
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Nhận diện đối tượng và vẽ bounding box
    results = model(image)
    annotated_image = results[0].plot()

    # Lưu ảnh đã vẽ bounding box vào đường dẫn tạm thời
    timestamp = int(time.time())
    temp_path = f'static/temp_{timestamp}.jpg'
    cv2.imwrite(temp_path, annotated_image)

    # return send_file(temp_path, mimetype='image/jpeg')
    return make_response(temp_path)
if __name__ == "__main__":
    app.run(debug=True , threaded=True)