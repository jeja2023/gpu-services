import asyncio
from collections import OrderedDict

from app.schemas import ModelBundle
from app.settings import GPU_QUEUE_LIMIT


MODEL_REGISTRY: "OrderedDict[str, ModelBundle]" = OrderedDict()
MODEL_LOAD_LOCKS: dict[str, asyncio.Lock] = {}
REGISTRY_LOCK = asyncio.Lock()
GPU_SEMAPHORE = asyncio.Semaphore(max(1, GPU_QUEUE_LIMIT))
