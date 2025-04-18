import logging

from flask import jsonify, request
from flask_restful import Resource

from thesis_code.thesis.webapp.src.backend.app._utils import APPApiResponse
from thesis_code.thesis.webapp.src.backend.app._iosamples import SAMPLE_INPUT
from thesis_code.thesis.webapp.src.backend.models.dart import Dart

logger = logging.getLogger(__name__)

class DartModel(Resource):
    """API for DART model"""

    def get(self):
        """GET method of DART model"""
        return jsonify({"SAMPLE_INPUT_FOR_POST_REQUEST": SAMPLE_INPUT})

    def post(self):
        """POST method of DART model"""
        prediction = Dart().get_forecast()
        return jsonify({
            "status": "success",
            "data": prediction
        })