from typing import Annotated, Optional

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.analysis_service import AnalysisService

_analysis_service: Optional[AnalysisService] = None


def get_analysis_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AnalysisService:
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService(settings)
    return _analysis_service


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
