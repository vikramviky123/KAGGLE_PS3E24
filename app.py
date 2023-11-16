from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from src.smoker_status.a_constants import *
from src.smoker_status.f_utils.common import load_pickle, read_yaml
from pathlib import Path


app = Flask(__name__)

candidates = {}


@app.route('/')
def index():
    return render_template('Intro.html')


@app.route('/data')
def data():
    return render_template('data.html')


@app.route('/eda/univariate')
def univariate():
    return render_template('univariate.html')


@app.route('/eda/bivariate')
def bivariate():
    return render_template('bivariate.html')


@app.route('/eda/multivariate')
def multivariate():
    return render_template('multivariate.html')


@app.route('/model/modelanalysis')
def modelanalysis():
    return render_template('modelanalysis.html')


def convert_to_float(value):
    return float(value) if value is not None and value != '' else None


@app.route('/model/modelplots')
def modelplots():
    return render_template('modelplots.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Retrieve the input data from the form
        age = request.form.get('age')
        height = request.form.get('height')
        weight = request.form.get('weight')
        waist = request.form.get('waist')
        eyesight_left = request.form.get('eyesight_left')
        eyesight_right = request.form.get('eyesight_right')
        hearing_left = request.form.get('hearing_left')
        hearing_right = request.form.get('hearing_right')
        systolic = request.form.get('systolic')
        relaxation = request.form.get('relaxation')
        fasting_blood_sugar = request.form.get('fasting_blood_sugar')
        cholesterol = request.form.get('cholesterol')
        triglyceride = request.form.get('triglyceride')
        hdl = request.form.get('hdl')
        ldl = request.form.get('ldl')
        hemoglobin = request.form.get('hemoglobin')
        urine_protein = request.form.get('urine_protein')
        serum_creatinine = request.form.get('serum_creatinine')
        ast = request.form.get('ast')
        alt = request.form.get('alt')
        gtp = request.form.get('gtp')
        dental_caries = request.form.get('dental_caries')

        # Load the saved list of models using pickle
        # Change this to the best model for blueberry yield
        best_model_name = 'lgbm_classifier'
        pickle_file_path = Path(
            "artifacts/model_trainer/trained_models.joblib")
        loaded_models = load_pickle(pickle_file_path)

        rf_models = loaded_models[best_model_name]

        # Convert form data to float
        def convert_to_float(value):
            return float(value) if value is not None and value != '' else None

        age = convert_to_float(age)
        height = convert_to_float(height)
        weight = convert_to_float(weight)
        waist = convert_to_float(waist)
        eyesight_left = convert_to_float(eyesight_left)
        eyesight_right = convert_to_float(eyesight_right)
        hearing_left = convert_to_float(hearing_left)
        hearing_right = convert_to_float(hearing_right)
        systolic = convert_to_float(systolic)
        relaxation = convert_to_float(relaxation)
        fasting_blood_sugar = convert_to_float(fasting_blood_sugar)
        cholesterol = convert_to_float(cholesterol)
        triglyceride = convert_to_float(triglyceride)
        hdl = convert_to_float(hdl)
        ldl = convert_to_float(ldl)
        hemoglobin = convert_to_float(hemoglobin)
        urine_protein = convert_to_float(urine_protein)
        serum_creatinine = convert_to_float(serum_creatinine)
        ast = convert_to_float(ast)
        alt = convert_to_float(alt)
        gtp = convert_to_float(gtp)
        dental_caries = convert_to_float(dental_caries)

        # Create a dictionary from the form data
        data = {
            'age': [age],
            'height': [height],
            'weight': [weight],
            'waist': [waist],
            'eyesight_left': [eyesight_left],
            'eyesight_right': [eyesight_right],
            'hearing_left': [hearing_left],
            'hearing_right': [hearing_right],
            'systolic': [systolic],
            'relaxation': [relaxation],
            'fasting_blood_sugar': [fasting_blood_sugar],
            'cholesterol': [cholesterol],
            'triglyceride': [triglyceride],
            'hdl': [hdl],
            'ldl': [ldl],
            'hemoglobin': [hemoglobin],
            'urine_protein': [urine_protein],
            'serum_creatinine': [serum_creatinine],
            'ast': [ast],
            'alt': [alt],
            'gtp': [gtp],
            'dental_caries': [dental_caries],

        }

        df = pd.DataFrame(data)
        preds = [model.predict_proba(np.array(df))[:, 1]
                 for model in rf_models]
        print(preds)
        preds_mean = np.mean(preds)
        if preds_mean > 0.43635852287376364:
            val_ = "Is a Smoker"
        else:
            val_ = "Not Smoker"

        return render_template('predict.html',
                               predicted_smoker=val_,
                               proba=preds_mean)

    except Exception as e:
        return render_template('predict.html', error_message=str(e))


@app.route('/form')
def show_form():
    return render_template('predict.html', preds_final=None, error_message=None)


if __name__ == '__main__':
    app.run(debug=True)
