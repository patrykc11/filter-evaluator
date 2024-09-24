from fastapi import HTTPException
from PIL import Image
from services.filters.cycle_gan import  process_with_cyclegan


def cycle_gan(image: Image):
    try:
        return process_with_cyclegan(image)
    except IOError:
        raise HTTPException(status_code=400, detail="Invalid image")