import logging

from flask import jsonify
from flask_restful import Resource

from thesis_code.thesis.webapp.src.backend.app._iosamples import SAMPLE_INPUT
from thesis_code.thesis.webapp.src.backend.models.linear_reg import LinRegModel

logger = logging.getLogger(__name__)

class LinearRegress(Resource):

    def get(self):
        return jsonify({"SAMPLE_INPUT_FOR_POST_REQUEST": SAMPLE_INPUT})

    def post(self):
        prediction = LinRegModel().get_forecast()  # Ensure this returns a dict
        print(prediction)
        # If prediction is accidentally a Response, uncomment:
        # if isinstance(prediction, Response):
        #     prediction = prediction.get_json()

        return jsonify({
            "status": "success",
            "data": prediction  # Must be a JSON-serializable dict/list
        })

