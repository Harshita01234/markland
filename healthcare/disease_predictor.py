import os
import re
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split

class DiseasePredictor:
    def __init__(self):
        # Get the directory of the current script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Possible paths to check
        possible_paths = [
            os.path.join(base_dir, 'Data', 'Training.csv'),
            os.path.join(base_dir, '..', 'Data', 'Training.csv'),
            os.path.join(base_dir, 'Training.csv'),
            os.path.join(base_dir, '..', 'Training.csv')
        ]
        
        # Find the first existing training data path
        training_path = self._find_existing_file(possible_paths, 'Training.csv')
        
        # Similar path checking for testing data
        possible_test_paths = [
            os.path.join(base_dir, 'Data', 'Testing.csv'),
            os.path.join(base_dir, '..', 'Data', 'Testing.csv'),
            os.path.join(base_dir, 'Testing.csv'),
            os.path.join(base_dir, '..', 'Testing.csv')
        ]
        testing_path = self._find_existing_file(possible_test_paths, 'Testing.csv')
        
        # Severity, description, and precaution paths
        severity_paths = [
            os.path.join(base_dir, 'MasterData', 'symptom_severity.csv'),
            os.path.join(base_dir, '..', 'MasterData', 'symptom_severity.csv')
        ]
        description_paths = [
            os.path.join(base_dir, 'MasterData', 'symptom_Description.csv'),
            os.path.join(base_dir, '..', 'MasterData', 'symptom_Description.csv')
        ]
        precaution_paths = [
            os.path.join(base_dir, 'MasterData', 'symptom_precaution.csv'),
            os.path.join(base_dir, '..', 'MasterData', 'symptom_precaution.csv')
        ]
        
        # Find paths for master data files
        severity_path = self._find_existing_file(severity_paths, 'symptom_severity.csv')
        description_path = self._find_existing_file(description_paths, 'symptom_Description.csv')
        precaution_path = self._find_existing_file(precaution_paths, 'symptom_precaution.csv')
        
        # Validate all required files exist
        self._validate_file_paths(training_path, testing_path, severity_path, 
                                  description_path, precaution_path)
        
        # Load training and testing data
        self.training = pd.read_csv(training_path)
        self.testing = pd.read_csv(testing_path)
        
        # Prepare columns and data
        self.cols = self.training.columns[:-1]
        self.x = self.training[self.cols]
        self.y = self.training['prognosis']
        
        # Encode labels
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.y)
        y_encoded = self.le.transform(self.y)
        
        # Split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, y_encoded, test_size=0.33, random_state=42
        )
        
        # Train Decision Tree Classifier
        self.clf = DecisionTreeClassifier()
        self.clf.fit(self.x_train, self.y_train)
        
        # Symptoms dictionary
        self.symptoms_dict = {symptom: index for index, symptom in enumerate(self.x)}
        
        # Load additional data
        self.severity_dictionary = self.load_csv(severity_path)
        self.description_list = self.load_csv(description_path, key_col=0, value_col=1)
        self.precaution_dictionary = self.load_precautions(precaution_path)
    
    def _find_existing_file(self, possible_paths, filename):
        """Find the first existing file from possible paths"""
        for path in possible_paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find {filename}. Tried paths: {possible_paths}")
    
    def _validate_file_paths(self, *paths):
        """Validate that all required files exist"""
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
    
    def load_csv(self, filepath, key_col=0, value_col=1):
        """Generic CSV loader for single-key dictionaries"""
        try:
            df = pd.read_csv(filepath)
            return dict(zip(df.iloc[:, key_col], df.iloc[:, value_col]))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return {}
    
    def load_precautions(self, filepath):
        """Load precautions from CSV"""
        try:
            df = pd.read_csv(filepath)
            return {row[0]: list(row[1:5]) for _, row in df.iterrows()}
        except Exception as e:
            print(f"Error loading precautions: {e}")
            return {}
    
    def predict_disease(self, symptoms, days):
        """Main disease prediction method"""
        # Prepare input vector
        input_vector = np.zeros(len(self.symptoms_dict))
        for symptom in symptoms:
            if symptom in self.symptoms_dict:
                input_vector[self.symptoms_dict[symptom]] = 1
        
        # Predict disease
        prediction = self.le.inverse_transform(
            self.clf.predict([input_vector])
        )[0]
        
        # Calculate condition severity
        severity_sum = sum(self.severity_dictionary.get(symptom, 0) for symptom in symptoms)
        severity_score = (severity_sum * days) / (len(symptoms) + 1)
        
        # Prepare results
        result = {
            'disease': prediction,
            'description': self.description_list.get(prediction, 'No description available'),
            'precautions': self.precaution_dictionary.get(prediction, []),
            'severity_warning': severity_score > 13
        }
        
        return result
    
    def get_all_symptoms(self):
        """Return list of all available symptoms"""
        return list(self.symptoms_dict.keys())

# Print current working directory and script location for debugging
print("Current Working Directory:", os.getcwd())
print("Script Location:", os.path.dirname(os.path.abspath(__file__)))