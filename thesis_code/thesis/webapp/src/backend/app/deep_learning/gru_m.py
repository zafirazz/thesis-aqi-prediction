from flask import jsonify
from flask_restful import Resource

from backend.app._iosamples import SAMPLE_INPUT
from backend.models.gru import Gru


class GruModel(Resource):
    """API of GRU model"""

    def get(self):
        """GET method of GRU model"""
        return jsonify({"SAMPLE_INPUT_FOR_POST_REQUEST": SAMPLE_INPUT})

    def post(self):
        """POST method of GRU model"""
        prediction = Gru().get_forecast()
        return jsonify({
            "status": "success",
            "data": prediction
        })