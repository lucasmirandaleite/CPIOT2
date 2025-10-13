
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import os
import cv2
import numpy as np

# Carregar o processador de imagem e o modelo pré-treinado
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Caminho para o dataset de imagens de teste
image_dir = "/home/ubuntu/tiny_coco_dataset/tiny_coco/val2017"
output_dir = "/home/ubuntu/huggingface_output"

os.makedirs(output_dir, exist_ok=True)

print(f"Iniciando detecção de objetos com Hugging Face (DETR) no diretório: {image_dir}")

# Lista para armazenar os resultados
results_summary = []

# Obter a lista de arquivos de imagem e processar apenas o primeiro
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith( (".jpg", ".jpeg", ".png") )]
if image_files:
    filename = image_files[0]
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path).convert("RGB")
    
    # Processar a imagem e fazer a inferência
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Converter saídas para o formato COCO (para avaliação)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    # Desenhar caixas delimitadoras e rótulos na imagem
    img_np = np.array(image)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        
        detected_objects.append({
            "class": label_name,
            "confidence": f"{round(score.item(), 2):.2f}",
            "bbox": [f"{coord:.2f}" for coord in box]
        })

        # Desenhar caixa e rótulo
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(img_cv2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Verde
        cv2.putText(img_cv2, f"{label_name}: {round(score.item(), 2):.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    output_path = os.path.join(output_dir, f"detected_{filename}")
    cv2.imwrite(output_path, img_cv2)

    results_summary.append({
        "image": filename,
        "detections": detected_objects,
        "num_detections": len(detected_objects)
    })
    
    print(f"Detectado {len(detected_objects)} objetos em {filename}. Imagem salva em {output_path}")
else:
    print(f"Nenhuma imagem encontrada no diretório: {image_dir}")

print("\nDetecção de objetos com Hugging Face (DETR) concluída.")
print("Resultados detalhados por imagem:")
for res in results_summary:
    print(f"- Imagem: {res['image']}")
    print(f"  Número de detecções: {res['num_detections']}")
    for det in res["detections"]:
        print(f"    - Objeto: {det['class']}, Confiança: {det['confidence']}, BBox: {det['bbox']}")

# Salvar o resumo em um arquivo para o README.md
with open("huggingface_results_summary.txt", "w") as f:
    f.write("### Resultados Hugging Face (DETR - Detecção de Objetos)\n\n")
    for res in results_summary:
        f.write(f"- **Imagem:** {res['image']}\n")
        f.write(f"  **Número de detecções:** {res['num_detections']}\n")
        if res["detections"]:
            f.write("  **Detecções:**\n")
            for det in res["detections"]:
                f.write(f"    - Objeto: {det['class']}, Confiança: {det['confidence']}, BBox: {det['bbox']}\n")
        else:
            f.write("  Nenhuma detecção.\n")
    f.write("\nAs imagens com as detecções foram salvas no diretório `/home/ubuntu/huggingface_output/`.")

