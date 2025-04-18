import logging

from flask import Flask
from flask_restful import Api

from backend.app.deep_learning.ensemble_m import EnsembleModel
from backend.app.deep_learning.gru_m import GruModel
from backend.app.deep_learning.lightgbm_m import LightGbmModel
from backend.app.deep_learning.lstm_v1_m import LstmModel
from backend.app.deep_learning.lstm_v2_m import LstmModel_reduced
from backend.app.download import DownloadReport
from backend.models.lstm_v1 import LstmOne
from thesis_code.thesis.webapp.src.backend.app.deep_learning.dart_m import DartModel
from thesis_code.thesis.webapp.src.backend.app.traditional_models.linear_m import LinearRegress
from thesis_code.thesis.webapp.src.backend.app.deep_learning.gbdt_m import GbdtModel

logger = logging.getLogger(__name__)


app = Flask(__name__)

api = Api(app)
api.add_resource(LinearRegress, "/api/model/linear_regression")
api.add_resource(LstmModel, "/api/model/lstm")
api.add_resource(GbdtModel, "/api/model/gbdt")
api.add_resource(DartModel, "/api/model/dart")
api.add_resource(LightGbmModel, "/api/model/lgbm")
api.add_resource(LstmModel_reduced, "/api/model/lstm_reduced")
api.add_resource(GruModel, "/api/model/gru")
api.add_resource(EnsembleModel, "/api/model/ensemble")
api.add_resource(DownloadReport, "/api/download_report")

@app.route("/api/check")
def api_check():
    return "all good"