from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # load a pretrained YOLO26n detection model
results = model("https://ultralytics.com/images/bus.jpg", save=True)  # predict and save the annotated image