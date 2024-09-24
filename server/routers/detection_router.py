from fastapi import APIRouter, UploadFile, File
from shared.common import image_to_buffer, read_image
from services.detection_service import detect_human
from services.filter_service import cycle_gan
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/detection",
    tags=["detection"],
    responses={404: {"description": "Not found"}},
)


@router.post("/run", tags=["detection"])
async def detect(file: UploadFile = File(...)):
    return StreamingResponse(image_to_buffer(detect_human(
        read_image(file))), media_type="image/jpeg")

@router.post("/with-filter/run", tags=["detection"])
async def detect_with_filter(file: UploadFile = File(...)):
    filtered_image = cycle_gan(read_image(file))
    return StreamingResponse(image_to_buffer(detect_human(
        filtered_image)), media_type="image/jpeg")
