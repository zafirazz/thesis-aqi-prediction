from flask import jsonify
from flask_restful import Resource

from backend.app._iosamples import SAMPLE_INPUT
from backend.models.gbdt import Gbdt


class GbdtModel(Resource):
    def get(self):
        return jsonify({"SAMPLE_INPUT_FOR_POST_REQUEST": SAMPLE_INPUT})

    def post(self):
        prediction = Gbdt().get_forecast()
        return jsonify({
            "status": "success",
            "data": prediction
        })