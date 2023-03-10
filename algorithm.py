import cv2
from ultralytics import YOLO

from draw import draw_box, draw_dot, draw_text
from project import socket

# Load a model
model = YOLO("yolov8n.pt")


def on_predict_batch_end(predictor):
    # results -> List[batch_size]
    _, _, im0s, _, _ = predictor.batch
    im0s = im0s if isinstance(im0s, list) else [im0s]
    predictor.results = zip(predictor.results, im0s)


ids = set()
counter = {}

model.add_callback("on_predict_batch_end", on_predict_batch_end)


loop = True


@socket.on("connect")
def connect():
    print("Client connected")
    global loop
    loop = True


@socket.on("disconnect")
def disconnect_handler():
    print("Client disconnected")
    ids.clear()
    counter.clear()
    global loop
    loop = False


@socket.event
def gen_frames(source: int | str = 0):
    for result, frame in model.track(
        source=source,
        stream=True,
        tracker="bytetrack.yaml",
        # classes=[2, 3, 5, 7, 8],
    ):
        if not loop:
            break
        for obj in result.boxes.boxes.cpu().numpy():
            #  check if client disconnected

            x1 = int(obj[0])
            y1 = int(obj[1])
            x2 = int(obj[2])
            y2 = int(obj[3])

            _id = int(obj[4])
            try:
                cat = int(obj[6])
            except IndexError:
                cat = 9999
            class_name = model.names[cat] if model.names else "Unknown"

            if _id not in ids:
                ids.add(_id)

                if class_name not in counter:
                    counter[class_name] = 0
                counter[class_name] += 1

            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            draw_box(frame, (x1, y1, x2, y2))
            draw_text(frame, f"{class_name}", (x1, y1 - 10))
            draw_dot(frame, (cx, cy), (0, 0, 255), 1)
        socket.emit("counter", counter)

        ret, _buffer = cv2.imencode(".jpg", frame)

        frame = _buffer.tobytes()
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )  # concat frame one by one and show results
