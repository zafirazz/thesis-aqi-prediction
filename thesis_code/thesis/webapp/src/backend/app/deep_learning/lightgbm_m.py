from flask import jsonify
from flask_restful import Resource

from backend.app._iosamples import SAMPLE_INPUT
from backend.models.gbdt import Gbdt
from backend.models.lightgbm import Lgbm


class LightGbmModel(Resource):
    """API of LGBM"""

    def get(self):
        """GET method of LGBM model"""
        return jsonify({"SAMPLE_INPUT_FOR_POST_REQUEST": SAMPLE_INPUT})

    def post(self):
        """POST method of LGBM model"""
        prediction = Lgbm().get_forecast()
        return jsonify({
            "status": "success",
            "data": prediction
        })