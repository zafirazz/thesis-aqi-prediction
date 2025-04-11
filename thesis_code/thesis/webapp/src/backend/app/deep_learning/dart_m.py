import logging

from flask import jsonify, request
from flask_restful import Resource

from thesis_code.thesis.webapp.src.backend.app._utils import APPApiResponse
from thesis_code.thesis.webapp.src.backend.app._iosamples import SAMPLE_INPUT
from thesis_code.thesis.webapp.src.backend.models.dart import Dart

logger = logging.getLogger(__name__)

class DartModel(Resource):
    def get(self):
        return jsonify({"SAMPLE_INPUT_FOR_DART": SAMPLE_INPUT})

    def post(self):
        payload = request.get_json(silent=True)
        if not payload:
            return APPApiResponse.fail(400, data="You did not provide a model name.")
        name = payload.get("model_name")
        prediction = Dart().get_forecast(name)
        return APPApiResponse.success(data=prediction)
