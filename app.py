from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_midterm.pkl'
model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)  # two ways to load the model, not using joblib here
#model = joblib.load('filename.pkl')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    MS = request.form['math_score']
    RS = request.form['reading_score']
    WS = request.form['writing_score']
    pred = model.predict(np.array([[MS, RS, WS]]))
    #print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(debug=True)
