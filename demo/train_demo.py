from ultralytics import YOLO

# 载入一个模型
model = YOLO('yolov8n-seg.yaml')  # 从YAML构建一个新模型
# model = YOLO('yolov8n-seg.pt')    # 载入预训练模型（推荐用于训练）
# model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # 从YAML构建并传递权重

# 训练模型
results = model.train(data='mydata.yaml', epochs=100, imgsz=640, batch=32)