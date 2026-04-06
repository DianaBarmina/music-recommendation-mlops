from fastapi import APIRouter, Depends

from services.api.dependencies import ModelArtifacts, get_artifacts

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(artifacts: ModelArtifacts = Depends(get_artifacts)):
    return {
        "status": "ok",
        "model_ready": artifacts.is_ready,
        "model_version": artifacts.model_version,
    }
