from fastapi import APIRouter
from .recommender import router as machine_learning_router
from .users import router as users_router

router = APIRouter(
    prefix="/api/v1"
)

router.include_router(machine_learning_router)
router.include_router(users_router)
