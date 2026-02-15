import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

from model import get_model

def load_model(model_path, num_classes, device):
    """Загрузка обученной модели"""
    model = get_model(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(model, image_path, class_names, device):
    """Предсказание для одного изображения"""
    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Загружаем и преобразуем изображение
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_idx = predicted.item()
    class_name = class_names[class_idx]
    confidence_score = confidence.item()
    
    # Топ-3 предсказания
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    top3_results = []
    for i in range(3):
        idx = top3_idx[0][i].item()
        prob = top3_prob[0][i].item()
        top3_results.append((class_names[idx], prob))
    
    return class_name, confidence_score, top3_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image for prediction')
    parser.add_argument('--class_names', type=str, required=True,
                       help='Path to class names file')
    args = parser.parse_args()
    
    # Загружаем имена классов
    with open(args.class_names, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем модель
    model = load_model(args.model_path, len(class_names), device)
    
    # Предсказание
    class_name, confidence, top3 = predict_image(
        model, args.image_path, class_names, device
    )
    
    print(f"\nPrediction Results:")
    print(f"Top prediction: {class_name}")
    print(f"Confidence: {confidence:.4f}")
    print("\nTop-3 predictions:")
    for i, (name, prob) in enumerate(top3, 1):
        print(f"{i}. {name}: {prob:.4f}")

if __name__ == '__main__':
    main()