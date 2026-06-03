from fastapi import APIRouter

from app.routes_debug import router as debug_router
from app.routes_health import router as health_router
from app.routes_models import router as models_router
from app.routes_person import router as person_router
from app.routes_vision import router as vision_router


router = APIRouter()
router.include_router(health_router)
router.include_router(models_router)
router.include_router(vision_router)
router.include_router(person_router)
router.include_router(debug_router)
