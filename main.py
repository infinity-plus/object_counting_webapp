import cv2
import torch

model = torch.hub.load(
    "ultralytics/yolov5",
    "yolov5s",
    pretrained=True,
)


#  Detect only vehicles
# model.classes = [1, 2, 3, 5, 7]

model.conf = 0.5


def draw_box(frame, box, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_line(frame, line, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = line
    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def draw_dot(frame, dot, color=(0, 255, 0), thickness=1):
    x, y = dot
    cv2.circle(frame, (x, y), 2, color, thickness)


def draw_text(frame, text, position, color=(0, 255, 0), thickness=1):
    x, y = position
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        thickness,
    )


def draw_counter(frame, count):
    draw_text(frame, str(count), (10, 50), (0, 255, 0), 2)


def draw_center(frame, box, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = box
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    draw_dot(frame, center, color, thickness)
    return center


X_MAX = 1020
Y_MAX = 500


def gen_frames():
    cap = cv2.VideoCapture(0)

    counter = 0
    count = 0
    offset = 6

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (X_MAX, Y_MAX))
        if not ret:
            break
        results = model(frame)
        counter += 1
        if counter % 4 != 0:
            continue

        for index, result in results.pandas().xyxy[0].iterrows():
            name = result["name"]

            x1 = int(result["xmin"])
            y1 = int(result["ymin"])
            x2 = int(result["xmax"])
            y2 = int(result["ymax"])
            draw_box(frame, (x1, y1, x2, y2))
            rectangle_center = draw_center(frame, (x1, y1, x2, y2))
            draw_text(frame, name, (x1, y1 - 10))
            cv2.line(
                frame, (0, (Y_MAX // 2) - 1), (X_MAX, (Y_MAX // 2)), (0, 0, 255), 2
            )

            #  check if the object crosses the center line
            if (
                rectangle_center[1] < 249 + offset
                and rectangle_center[1] > 249 - offset
            ):
                count += 1
        draw_counter(frame, count)
        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )  # concat frame one by one and show resultss
