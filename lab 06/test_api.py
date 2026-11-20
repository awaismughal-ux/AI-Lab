"""
Unit Test Suite for Lab 06: Model Deployment & API Integration
Course: Introduction to Artificial Intelligence (AI-101L)

Verifies:
1. Model deserialization integrity
2. Correct feature ordering and input handling
3. Low-risk vs high-risk clinical inference behavior
4. Output schema compliance
"""

import sys
import unittest
from pathlib import Path
from main import make_prediction, FEATURES, model


class TestDiabetesAPI(unittest.TestCase):

    def test_model_loaded(self):
        """Verify model object exists and has required attributes."""
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))
        self.assertEqual(len(FEATURES), 6)

    def test_low_risk_prediction(self):
        """Verify inference on a healthy profile yields Low Risk / Non-Diabetic."""
        healthy_vitals = {
            "Glucose": 85.0,
            "BloodPressure": 66.0,
            "Insulin": 45.0,
            "BMI": 22.4,
            "DiabetesPedigreeFunction": 0.18,
            "Age": 25.0
        }
        res = make_prediction(healthy_vitals)
        self.assertIn("predicted_label", res)
        self.assertIn("probability", res)
        self.assertIn("risk_level", res)
        self.assertEqual(res["predicted_label"], 0)
        self.assertEqual(res["prediction"], "Not Diabetic")
        self.assertEqual(res["risk_level"], "Low Risk")
        self.assertLess(res["probability"], 0.35)

    def test_high_risk_prediction(self):
        """Verify inference on an acute diabetic profile yields High Risk / Diabetic."""
        diabetic_vitals = {
            "Glucose": 185.0,
            "BloodPressure": 90.0,
            "Insulin": 250.0,
            "BMI": 39.5,
            "DiabetesPedigreeFunction": 0.95,
            "Age": 55.0
        }
        res = make_prediction(diabetic_vitals)
        self.assertEqual(res["predicted_label"], 1)
        self.assertEqual(res["prediction"], "Diabetic")
        self.assertEqual(res["risk_level"], "High Risk")
        self.assertGreater(res["probability"], 0.65)

    def test_probability_bounds(self):
        """Ensure predicted probabilities are strictly bounded within [0, 1]."""
        test_patient = {
            "Glucose": 120.0,
            "BloodPressure": 75.0,
            "Insulin": 100.0,
            "BMI": 28.0,
            "DiabetesPedigreeFunction": 0.40,
            "Age": 38.0
        }
        res = make_prediction(test_patient)
        self.assertTrue(0.0 <= res["probability"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
