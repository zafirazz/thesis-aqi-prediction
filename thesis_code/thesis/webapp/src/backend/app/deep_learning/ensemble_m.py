from flask import jsonify
from flask_restful import Resource

from backend.app._iosamples import SAMPLE_INPUT
from backend.models.ensemble_model import Ensemble


class EnsembleModel(Resource):
    """Ensemble model API"""

    def get(self):
        """GET method API EnsembleModel"""
        return jsonify({"SAMPLE_INPUT_FOR_POST_REQUEST": SAMPLE_INPUT})

    def post(self):
        """POST method API EnsembleModel"""
        prediction = Ensemble().get_forecast()
        return jsonify({
            "status": "success",
            "data": prediction
        })