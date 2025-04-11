from flask import jsonify
from flask_restful import Resource

from backend.data_load.get_stats import GetStats


class DownloadReport(Resource):

    def get(self):
        return jsonify({"Hello there, this is get request."})

    def post(self):
        stats = GetStats().stats_data()
        return jsonify({
            "status": "success",
            "data": stats,
        })