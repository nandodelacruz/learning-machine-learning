from ultralytics import YOLO


def print_box(box):
    class_id, cords, conf = box
    print("Object type:", class_id)
    print("Coordinates:", cords)
    print("Probability:", conf)
    print("---")


def main():
    model = YOLO('yolov8m.pt')

    results = model.predict('cat_dog.png')

    for result in results:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            print_box((class_id, cords, conf))


if __name__ == "__main__":
    main()