from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_HW3_mdl3.pkl'
model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename) 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    FA = request.form['Fixed_Acidity']
    VA = request.form['Volatile_Acidity']
    CA = request.form['Citric_Acid']
    RS = request.form['Residual_Sugar']
    CL = request.form['Chlorides']
    FS = request.form['Free_Sulpher_Dioxide']
    DS = request.form['Density']
    PH = request.form['PH']
    SP = request.form['Sulphates']
    AL = request.form['Alcohol']
    pred = model.predict(np.array([[FA, VA, CA, RS, CL, FS, DS, PH , SP, AL]]))
    #print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(debug=True)