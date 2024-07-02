import tensorflow as tf
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Define transformation to convert image to tensor and back
transform = transforms.Compose([
    transforms.Resize((256, 512), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

inv_transform = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2)),
    transforms.ToPILImage(),
])

# Load TensorFlow model
def load_tf_model(checkpoint_dir):
    tf.compat.v1.disable_eager_execution()  # Wyłącz tryb eager execution
    sess = tf.compat.v1.Session()
    
    # Load the meta graph and restore weights
    saver = tf.compat.v1.train.import_meta_graph(f'{checkpoint_dir}/cyclegan.model-106002.meta')
    try:
        saver.restore(sess, f'{checkpoint_dir}/cyclegan.model-106002')
        print("Model restored successfully from checkpoint.")
    except Exception as e:
        print(f"Failed to restore model: {e}")
        return None, None
    
    # Get the default graph
    graph = tf.compat.v1.get_default_graph()

    # Diagnostic: Check if graph is loaded
    if not graph.get_operations():
        print("Graph is empty. Failed to load the model.")
        return None, None
    else:
        print(f"Graph loaded with {len(graph.get_operations())} operations.")

    # Diagnostic: Check for key operations (e.g., input and output tensors)
    try:
        input_tensor = graph.get_tensor_by_name('test_A:0')
        output_tensor = graph.get_tensor_by_name('test_B:0')
        print("Key tensors found in the graph.")
    except KeyError as e:
        print(f"Key tensor not found in the graph: {e}")
        return None, None
    
    return graph, sess

def list_operations(graph, filename):
    with open(filename, 'w') as f:
        for op in graph.get_operations():
            f.write(op.name + '\n')

checkpoint_dir = 'pretrained'
graph, sess = load_tf_model(checkpoint_dir)

# Save the list of operations to a file
# list_operations(graph, 'operations_list.txt')

input_tensor_name = 'test_A:0'  # Adjust the tensor name as needed
output_tensor_name = 'test_B:0'  # Adjust the tensor name as needed

def process_tile_tf(tile, graph, sess):
    # Ensure the image is in RGB mode
    tile = tile.convert('RGB')
    
    # Transform image to tensor
    tensor = transform(tile).unsqueeze(0).numpy()
    tensor = np.transpose(tensor, (0, 2, 3, 1))
    tensor = np.tile(tensor, (4, 1, 1, 1))

    # Get input and output tensors
    input_tensor = graph.get_tensor_by_name(input_tensor_name)  # Adjust the tensor name as needed
    output_tensor = graph.get_tensor_by_name(output_tensor_name)  # Adjust the tensor name as needed

    # Run the model
    output = sess.run(output_tensor, feed_dict={input_tensor: tensor})
    
    # Convert output tensor to image
    output_image = inv_transform(torch.tensor(np.transpose(output[0], (2, 0, 1))))
    return output_image

def split_image(image, tile_size):
    width, height = image.size
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tiles.append((tile, x, y))
    return tiles

def merge_tiles(tiles, image_size):
    width, height = image_size
    new_image = Image.new('RGB', (width, height))
    for tile, x, y in tiles:
        new_image.paste(tile, (x, y))
    return new_image

# Load and process the image
input_image = Image.open('../images/original_day/image_1.jpg')
tiles = split_image(input_image, 256)
processed_tiles = [(process_tile_tf(tile, graph, sess), x, y) for tile, x, y in tiles]
output_image = merge_tiles(processed_tiles, input_image.size)

# Save or display the output image
output_image.save('output_image.jpg')
output_image.show()