from flask import Flask,render_template,request
import cv2
from tensorflow.keras.models import load_model#to load our trained model
import numpy as np
from werkzeug.utils import secure_filename
app = Flask(__name__,template_folder="templates") # initializing a flask app
# Loading the model
model=load_model('model.h5')
print("Loaded model from disk")
@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')
@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')
@app.route('/intro', methods=['GET'])
def about():
    return render_template('intro.html')

def predict():

        # Get a reference to webcam #0 (the default one)

        print("[INFO] starting video stream...")
        vs = cv2.VideoCapture(0)
        #writer = None
        (W, H) = (None, None)

# loop over frames from the video file stream
        while True:
        	# read the next frame from the file
            (grabbed, frame) = vs.read()

        	# if the frame was not grabbed, then we have reached the end
        	# of the stream
            if not grabbed:
                break
            if W is None or H is None:
                (H, W) = frame.shape[:2]

        	# clone the output frame, then convert it from BGR to RGB
        	# ordering and resize the frame to a fixed 64x64
            output = frame.copy()
            #print("apple")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (64, 64))
            #frame = frame.astype("float32")
            x=np.expand_dims(frame, axis=0)
            result = np.argmax(model.predict(x), axis=-1)
            index=['Cyclone','Earthquake','Flood','Wildfire']
            result=str(index[result[0]])
            #print(result)
            #result=result.tolist()

            cv2.putText(output, "activity: {}".format(result), (10, 120), cv2.FONT_HERSHEY_PLAIN,
                        1, (0,255,255), 1)
            #playaudio("Emergency it is a disaster")
            cv2.imshow("Output", output)
            key = cv2.waitKey(1) & 0xFF

        		# if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # release the file pointers
        print("[INFO] cleaning up...")
        vs.release()
        cv2.destroyAllWindows()
        return render_template("upload.html")

if __name__ == '_main_':
      app.run(host='0.0.0.0', port=8000,debug=False)