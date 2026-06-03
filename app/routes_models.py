from fastapi import APIRouter

from app.routes_model_lifecycle import router as lifecycle_router
from app.routes_model_query import router as query_router
from app.routes_predict import router as predict_router
from app.routes_rollout import router as rollout_router

router = APIRouter()
router.include_router(query_router)
router.include_router(lifecycle_router)
router.include_router(predict_router)
router.include_router(rollout_router)
