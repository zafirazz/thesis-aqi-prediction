import logging

from flask import jsonify
from flask_restful import Resource

from backend.app._iosamples import SAMPLE_INPUT
from backend.models.linear_reg import LinRegModel
from backend.models.lstm_v1 import LstmOne

logger = logging.getLogger(__name__)

class LstmModel(Resource):
    """API for LSTM model with 4 layers"""

    def get(self):
        """GET method of LSTM (4 layers)"""
        return jsonify({"SAMPLE_INPUT_FOR_POST_REQUEST": SAMPLE_INPUT})

    def post(self):
        """POST method of LSTM (4 layers)"""
        prediction = LstmOne().get_forecast()
        return jsonify({
            "status": "success",
            "data": prediction
        })