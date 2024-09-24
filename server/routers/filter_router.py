from fastapi import APIRouter, UploadFile, File
from services.filter_service import cycle_gan
from shared.common import image_to_buffer, read_image
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/filter",
    tags=["filter"],
    responses={404: {"description": "Not found"}},
)


@router.post("/cycle-gan/run", tags=["filter"])
async def filter_image_with_cycle_gan(file: UploadFile = File(...)):
    return StreamingResponse(image_to_buffer(cycle_gan(
        read_image(file))), media_type="image/jpeg")

