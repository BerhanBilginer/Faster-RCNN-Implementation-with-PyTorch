import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import requests

# Load your custom model
model_path = 'model/model_standart_values_2.torchscript'
#model = torch.load(model_path, map_location=torch.device('cpu'))
model = torch.jit.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Define the image transformation
transform = T.Compose([T.ToTensor()])

def detect_objects(model, image_path):
    # Load and transform the input image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model((image_tensor))

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    print(predictions.shape)
    for box in predictions[0]['boxes']:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

    # Display the image with bounding boxes
    image.show()

# Replace 'your_image_path.jpg' with the path to your test image
image_path = "/home/berhan/Desktop/Development-Berhan/Test_Workspace/dataset/output_dataset/test/images/000621_jpg.rf.3b2dbf0f4eccb61a010720dd6fd078cb.jpg"
detect_objects(model, image_path)
