import stone
from json import dumps
import numpy as np
import math
from PIL import ImageColor
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
from timeit import default_timer

class HairSegmentModel(nn.Module):
    def __init__(self):
        super(HairSegmentModel,self).__init__()
        deeplab = torchvision.models.segmentation.deeplabv3_resnet50(weights=0, progress=1, num_classes=2)
        self.dl = deeplab
        
    def forward(self, x):
        y = self.dl(x)['out']
        return y

def personal_color(skin_rgb, hair_rgb):
    """
    Xác định nhóm màu cá nhân dựa trên khoảng cách Euclidean
    từ (skin_rgb, hair_rgb) đến tâm của từng nhóm màu.
    Luôn trả về nhóm gần nhất — không bao giờ trả về None.
    """
    # Tâm (centroid) của mỗi nhóm màu: (skin_R, skin_G, skin_B, hair_R, hair_G, hair_B)
    CENTROIDS = {
        "Warm Spring":  (243.852, 202.359, 180.464, 108.651,  83.833,  70.887),
        "Light Spring": (243.986, 226.830, 215.576, 214.960, 204.645, 204.885),
        "Clear Spring": (227.570, 174.053, 143.618,  35.910,  31.950,  38.027),
        "Light Summer": (231.460, 184.660, 165.630, 158.000,  98.157,  81.982),
        "Soft Summer":  (226.530, 206.212, 163.026,  92.924,  78.635,  72.345),
        "Cool Summer":  (203.620, 152.978, 126.393,  17.667,  18.890,  19.287),
        "Soft Autumn":  (222.086, 177.840, 145.870, 123.920,  90.707,  72.247),
        "Deep Autumn":  (203.960, 147.296, 113.128,  44.397,  35.099,  26.589),
        "Warm Autumn":  (179.868, 128.390, 100.100, 142.795,  91.614,  73.373),
        "Deep Winter":  (226.107, 171.970, 144.297,  17.667,  18.890,  19.287),
        "Clear Winter": (245.454, 207.065, 191.890, 110.542,  90.260,  77.610),
        "Cool Winter":  (222.420, 146.424, 167.488,  54.000,  37.069,  34.116),
    }

    skin_r, skin_g, skin_b = skin_rgb
    hair_r, hair_g, hair_b = hair_rgb
    input_vec = (skin_r, skin_g, skin_b, hair_r, hair_g, hair_b)

    best_color = "Light Summer"  # fallback an toàn
    best_dist  = float('inf')

    for color_name, centroid in CENTROIDS.items():
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(input_vec, centroid)))
        print(f'[DEBUG] {color_name}: distance = {dist:.2f}')
        if dist < best_dist:
            best_dist  = dist
            best_color = color_name

    print(f'[DEBUG] Personal color result: {best_color} (dist={best_dist:.2f})')
    return best_color

def detect_face(image_path):
    # Load the pre-trained Haar cascade file for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        return True
    else:
        return False

def get_skin_color(image_path):
    result = stone.process(image_path, image_type='auto', n_dominant_colors=1, return_report_image=True)
    report_images = result.pop("report_images") 
    face_list = [id for id in report_images.keys()]
    face_id = face_list[0]
    # Uncomment the line below to show image with measurements
    # stone.show(report_images[face_id])  

    result_json = dumps(result)
    results = result_json.split(',')
    skin_tone = "#000000"
    for item in results:
        if "dominant" in item:
            skin_tone = item.split(':')[-1].replace('\"','').strip()

    hex_code = skin_tone.lstrip('#')
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return r, g, b

def get_hair_mask(image_path, checkpoint_path="D:/hair_detect.pt"):
    if isinstance(image_path, np.ndarray):
        img = Image.fromarray(image_path)
    else:
        img = Image.open(image_path)
    
    preprocess = transforms.Compose([transforms.Resize((512, 512), 2),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    Xtest = preprocess(img)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = HairSegmentModel()
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        model.eval()
        device = torch.device('cpu') # cpu | cuda
        model.to(device)
        Xtest = Xtest.to(device).float()
        ytest = model(Xtest.unsqueeze(0).float())
        ypos = ytest[0, 1, :, :].clone().detach().cpu().numpy()
        yneg = ytest[0, 0, :, :].clone().detach().cpu().numpy()
        ytest = ypos >= yneg
    
    mask = ytest.astype('float32')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.dilate(mask,kernel,iterations = 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def get_hair_color(image_path):
    image = cv2.imread(image_path)  
    if image is None:
        print("Failed to load the image from:", image_path)
        return
    if image.shape[0] < 1 or image.shape[1] < 1:
        print("Invalid image dimensions:", image.shape)
        return
    image = cv2.resize(image, (512, 512))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = get_hair_mask(rgb)
    bool_mask = mask.astype(bool)
    region = image[bool_mask]

    hair_color = []
    for c in range(3):
        unique, counts = np.unique(region[:,c], return_counts=True)
        hair_color.append(unique[counts.argmax()])

    hair_color_rgb = hair_color[::-1]
    return tuple(hair_color_rgb)
