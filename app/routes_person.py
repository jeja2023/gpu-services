from fastapi import APIRouter

from app.routes_person_detection import router as detection_router
from app.routes_person_embeddings import router as embeddings_router
from app.routes_person_stream import router as stream_router
from app.routes_person_tracks import router as tracks_router
from app.routes_person_video import router as video_router


router = APIRouter()
router.include_router(detection_router)
router.include_router(embeddings_router)
router.include_router(tracks_router)
router.include_router(video_router)
router.include_router(stream_router)
