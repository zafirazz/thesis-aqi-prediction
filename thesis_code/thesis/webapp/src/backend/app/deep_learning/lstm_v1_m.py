import logging

from flask import jsonify
from flask_restful import Resource

from backend.app._iosamples import SAMPLE_INPUT
from backend.models.linear_reg import LinRegModel
from backend.models.lstm_v1 import LstmOne

logger = logging.getLogger(__name__)

class LstmModel(Resource):
    def get(self):
        return jsonify({"SAMPLE_INPUT_FOR_POST_REQUEST": SAMPLE_INPUT})

    def post(self):
        prediction = LstmOne().get_forecast()
        return jsonify({
            "status": "success",
            "data": prediction
        })