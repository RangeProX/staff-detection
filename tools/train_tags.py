from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="dataset/tag.yaml", epochs=30, imgsz=960,
            degrees=180, flipud=0.5, fliplr=0.5, hsv_v=0.4, patience=30,
            name="tag_v4")