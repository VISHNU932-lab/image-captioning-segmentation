import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import cv2

def _get_device(device):
    if device == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        return torch.device('cuda')

def load_segmentation_model(device='cpu'):
    DEVICE = _get_device(device)
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.to(DEVICE)
    model.eval()
    return model

def load_caption_model(device='cpu'):
    DEVICE = _get_device(device)
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
    model.to(DEVICE)
    model.eval()
    return processor, model

def preprocess_for_maskrcnn(pil_img: Image.Image):
    transform = T.Compose([T.ToTensor()])
    return transform(pil_img)

@torch.no_grad()
def run_segmentation(model, pil_img: Image.Image, score_thresh=0.5):
    DEVICE = next(model.parameters()).device
    img_tensor = preprocess_for_maskrcnn(pil_img).to(DEVICE)
    outputs = model([img_tensor])[0]
    masks = outputs.get('masks')
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    selected = scores >= score_thresh
    if masks is None or len(masks)==0:
        return [], labels[selected].tolist(), scores[selected].tolist()
    masks = masks[selected].cpu().numpy()
    bool_masks = []
    for m in masks:
        mask = m[0] > 0.5
        bool_masks.append(mask)
    return bool_masks, labels[selected].tolist(), scores[selected].tolist()

def overlay_masks(pil_img: Image.Image, masks, alpha=0.5):
    img = np.array(pil_img.convert('RGBA')).astype(np.uint8)
    h, w = img.shape[:2]
    overlay = img.copy()
    for mask in masks:
        color = np.random.randint(0, 255, size=(3,))
        color_layer = np.zeros((h,w,4), dtype=np.uint8)
        color_layer[...,:3] = color
        color_layer[...,3] = (mask.astype(np.uint8) * int(255*alpha))
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)
    return Image.fromarray(overlay)

@torch.no_grad()
def generate_caption(processor, model, pil_img: Image.Image, max_length=30):
    DEVICE = next(model.parameters()).device
    inputs = processor(images=pil_img, return_tensors='pt').to(DEVICE)
    out = model.generate(**inputs, max_length=max_length)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
