from flask import jsonify
from flask_restful import Resource

from backend.data_load.get_stats import GetStats


class DownloadReport(Resource):
    """API for downloading excel report."""
    def get(self):
        """Get method of API"""
        return jsonify({"Hello there, this is get request."})

    def post(self):
        """Post method of API"""
        stats = GetStats().stats_data()
        return jsonify({
            "status": "success",
            "data": stats,
        })