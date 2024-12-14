from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import the DiseasePredictor
from disease_predictor import DiseasePredictor

app = Flask(__name__)
predictor = DiseasePredictor()

@app.route('/')
def index():
    symptoms = predictor.get_all_symptoms()
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        selected_symptoms = request.form.getlist('symptoms')
        days = int(request.form.get('days', 1))
        
        # Validate inputs
        if not selected_symptoms:
            return jsonify({
                'error': 'Please select at least one symptom.'
            }), 400
        
        # Predict disease
        result = predictor.predict_disease(selected_symptoms, days)
        
        return render_template('result.html', 
                               disease=result['disease'],
                               description=result['description'],
                               precautions=result['precautions'],
                               severity_warning=result['severity_warning'])
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)