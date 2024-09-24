import os
from PIL import Image, ImageDraw, ImageFont
import shutil

def create_image_grid_for_all_images(main_folder, other_folders, output_folder):
    image_names = [f for f in os.listdir(main_folder) if f.endswith(('png', 'jpg', 'jpeg', 'bmp'))]

    for image_name in image_names:
        images = []
        labels = []

        for folder in other_folders:
            image_path = os.path.join(folder, image_name)

            if os.path.exists(image_path):
                image = Image.open(image_path)
                images.append(image)
                labels.append(os.path.basename(folder))
            else:
                print(f"Image {image_name} not found in folder {folder}")
                continue

        if images:
            target_size = images[0].size 
            images = [img.resize(target_size) for img in images] 

            font_size = 20
            font = ImageFont.load_default()

            img_width, img_height = target_size
            grid_cols = 3
            grid_rows = 4

            total_width = grid_cols * img_width
            total_height = grid_rows * (img_height + font_size + 10) 

            new_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
            draw = ImageDraw.Draw(new_image)

            for index, image in enumerate(images):
                x_offset = (index % grid_cols) * img_width
                y_offset = (index // grid_cols) * (img_height + font_size + 10)

                new_image.paste(image, (x_offset, y_offset))

                label = labels[index]
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = x_offset + (img_width - text_width) // 2
                text_y = y_offset + img_height + 5

                draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

            output_path = os.path.join(output_folder, f"grid_{image_name}")
            new_image.save(output_path)
            print(f"Saved: {output_path}")

camera_names = ["caorle", "hungary_papa", "iseo_garibaldi"] #"brugia",

for name in camera_names:

    main_folder = f"./images/images_{name}"

    other_folders = [
        f"./images/images_{name}_hog",
        f"./images/images_{name}_r_cnn",
        f"./images/images_{name}_yolo_8x",
        f"./images/custom_filter_images_{name}_hog",
        f"./images/custom_filter_images_{name}_r_cnn",
        f"./images/custom_filter_images_{name}_yolo_8x",
        f"./images/cycle_gan_images_{name}_hog",
        f"./images/cycle_gan_images_{name}_r_cnn",
        f"./images/cycle_gan_images_{name}_yolo_8x",
        f"./images/sid_images_{name}_hog",
        f"./images/sid_images_{name}_r_cnn",
        f"./images/sid_images_{name}_yolo_8x",
    ]

    output_folder = f"./images/grids/{name}"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    create_image_grid_for_all_images(main_folder, other_folders, output_folder)