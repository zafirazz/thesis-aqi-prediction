import logging
import traceback
from functools import wraps
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

class APPApiResponse:
    """Air Pollution Prediction app API response."""

    @staticmethod
    def fail(status_code: int, data: Union[str, Any]):
        """Fail."""
        return {"status": "fail", "Data": data}, status_code

    @staticmethod
    def error(status_code: int = 418, msg: Optional[str] = None, data: Optional[str] = None):
        """Error."""
        logger.error(msg, exc_info=True)
        return (
            {"status": "error", "message": msg or 'Something is wrong', "data": data or traceback.format_exc()},
            status_code,
        )

    @staticmethod
    def success(status_code: int = 200, data: Any = None):
        """Success."""
        print(f"data from utils {data}")
        return {"status": "success", "data": data}, status_code

    @staticmethod
    def error_handler(func):
        """Error handler decorator."""
        @wraps
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return APPApiResponse.error()
        return wrapper