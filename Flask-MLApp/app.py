import numpy as np
from flask import Flask, request, jsonify, render_template

from sklearn.externals import joblib

app = Flask(__name__)
model = joblib.load('customer_churn_mlmodel.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
	
    if (prediction[0] == 1):
        output = 'Customer will be leave from bank So please give offer'
    else:
        output = 'Customer will be not leave from bank'
    return render_template('index.html', prediction_text=output)


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

#if __name__ == "__main__":
#    app.run(host='0.0.0.0',port=8080)
