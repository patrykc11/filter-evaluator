from PIL import Image, ImageDraw, ImageFont

def create_image_grid(images, captions, output_path, img_size=(300, 300)):
    width, height = img_size
    grid_width = 2 * width
    grid_height = 2 * height + 40

    grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default() 

    draw = ImageDraw.Draw(grid_image)

    positions = [(0, 0), (width, 0), (0, height), (width, height)]
    for i, (img, caption, position) in enumerate(zip(images, captions, positions)):

        img = img.resize(img_size)
        grid_image.paste(img, position)

        text_position = (position[0] + 10, position[1] + height + 10) 
        draw.text(text_position, caption, fill=(0, 0, 0), font=font)

    grid_image.save(output_path)

if __name__ == "__main__":

    image_paths = [
        "images/images_brugia/frame_0002.png",
        "images/cycle_gan_images_brugia/frame_0002.png",
        "images/sid_images_brugia/frame_0002.png",
        "images/custom_filter_images_brugia/frame_0002.png"
    ]
    images = [Image.open(image) for image in image_paths]

    captions = ["Oryginalny", "CycleGAN", "SID", "Custom filter"]

    output_image = "grid_image.png"

    create_image_grid(images, captions, output_image, img_size=(300, 300))