from fastapi import FastAPI
from routers import detection_router, filter_router

app = FastAPI()

app.include_router(detection_router.router)
app.include_router(filter_router.router)