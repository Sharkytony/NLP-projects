from flask import Flask, render_template, request
import joblib

app = Flask(__name__, static_folder='static')

loaded_preprocessor = joblib.load('preprocessor_pipeline.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    loaded_model = joblib.load('classifier_model.pkl')
    email_text = request.form['emailText']
    preprocess = loaded_preprocessor.transform(email_text)
    pred = loaded_model.predict(preprocess)
    if pred == 1:
        return render_template('spam_page.html')
    else:
        return render_template('not_spam_page.html')

if __name__ == '__main__':
    app.run(debug=True)