import logging

from flask import jsonify
from flask_restful import Resource

from backend.app._iosamples import SAMPLE_INPUT
from backend.models.lstm_v2 import LstmTwo

logger = logging.getLogger(__name__)

class LstmModel_reduced(Resource):
    """API for LSTM with 2 layers"""

    def get(self):
        """GET method of LSTM reduced layers"""
        return jsonify({"SAMPLE_INPUT_FOR_POST_REQUEST": SAMPLE_INPUT})

    def post(self):
        """POST method of LSTM reduced layers"""
        prediction = LstmTwo().get_forecast()
        return jsonify({
            "status": "success",
            "data": prediction
        })