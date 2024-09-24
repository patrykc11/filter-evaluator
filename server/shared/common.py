from fastapi import UploadFile, HTTPException
from PIL import Image
import io

valid_content_types = ["image/jpeg", "image/png", "image/bmp"]


def read_image(file: UploadFile):
    if file.content_type not in valid_content_types:
        raise HTTPException(status_code=400, detail="Invalid file type: " +
                            f"{file.filename}. Please upload an image file.")
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process" +
                            f"the image file: {file.filename}")


def image_to_buffer(image: Image):
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    buf.seek(0)

    return buf
