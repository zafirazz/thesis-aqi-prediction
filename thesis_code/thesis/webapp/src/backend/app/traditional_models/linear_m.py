import logging

from flask import jsonify
from flask_restful import Resource
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from thesis_code.thesis.webapp.src.backend.app._iosamples import SAMPLE_INPUT
from thesis_code.thesis.webapp.src.backend.models.linear_reg import LinRegModel

logger = logging.getLogger(__name__)

class LinearRegress(Resource):
    """API for Linear Regression model"""

    def get(self):
        """GET method of API"""
        return jsonify({"SAMPLE_INPUT_FOR_POST_REQUEST": SAMPLE_INPUT})

    def post(self):
        """POST method of API"""
        prediction = LinRegModel(LinearRegression()).get_forecast()
        return jsonify({
            "status": "success",
            "data": prediction
        })

