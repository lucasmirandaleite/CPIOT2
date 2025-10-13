
from ultralytics import YOLO
import os
import cv2

# Carregar um modelo YOLOv8 pré-treinado
# Usaremos o modelo nano para demonstração rápida
model = YOLO("yolov8n.pt")

# Caminho para o dataset de imagens de teste
# O tiny_coco_dataset tem uma estrutura de pastas, vamos usar as imagens de teste
image_dir = "/home/ubuntu/tiny_coco_dataset/tiny_coco/val2017"
output_dir = "/home/ubuntu/yolov8_output"

os.makedirs(output_dir, exist_ok=True)

print(f"Iniciando detecção de objetos com YOLOv8 no diretório: {image_dir}")

# Lista para armazenar os resultados
results_summary = []

# Iterar sobre as imagens no diretório
for filename in os.listdir(image_dir):
    if filename.lower().endswith( (".jpg", ".jpeg", ".png") ):
        image_path = os.path.join(image_dir, filename)
        
        # Realizar a detecção
        results = model(image_path, save=False, verbose=False) # save=False para não salvar imagens automaticamente
        
        # Processar os resultados
        for r in results:
            im_bgr = r.plot() # Plota as caixas delimitadoras e labels na imagem
            output_path = os.path.join(output_dir, f"detected_{filename}")
            cv2.imwrite(output_path, im_bgr)
            
            # Coletar informações sobre as detecções
            boxes = r.boxes.xyxy.tolist() # Coordenadas das caixas
            classes = r.boxes.cls.tolist() # IDs das classes
            names = r.names # Mapeamento de ID para nome da classe
            confidences = r.boxes.conf.tolist() # Confiança das detecções
            
            detected_objects = []
            for box, cls, conf in zip(boxes, classes, confidences):
                detected_objects.append({
                    "class": names[int(cls)],
                    "confidence": f"{conf:.2f}",
                    "bbox": [f"{coord:.2f}" for coord in box]
                })
            
            results_summary.append({
                "image": filename,
                "detections": detected_objects,
                "num_detections": len(detected_objects)
            })
            
            print(f"Detectado {len(detected_objects)} objetos em {filename}. Imagem salva em {output_path}")

print("\nDetecção de objetos com YOLOv8 concluída.")
print("Resultados detalhados por imagem:")
for res in results_summary:
    print(f"- Imagem: {res['image']}")
    print(f"  Número de detecções: {res['num_detections']}")
    for det in res["detections"]:
        print(f"    - Objeto: {det['class']}, Confiança: {det['confidence']}, BBox: {det['bbox']}")

# Salvar o resumo em um arquivo para o README.md
with open("yolov8_results_summary.txt", "w") as f:
    f.write("### Resultados YOLOv8 (Detecção de Objetos)\n\n")
    for res in results_summary:
        f.write(f"- **Imagem:** {res['image']}\n")
        f.write(f"  **Número de detecções:** {res['num_detections']}\n")
        if res["detections"]:
            f.write("  **Detecções:**\n")
            for det in res["detections"]:
                f.write(f"    - Objeto: {det['class']}, Confiança: {det['confidence']}, BBox: {det['bbox']}\n")
        else:
            f.write("  Nenhuma detecção.\n")
    f.write("\nAs imagens com as detecções foram salvas no diretório `/home/ubuntu/yolov8_output/`.")

