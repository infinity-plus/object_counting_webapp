import cv2


def draw_box(frame, box, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


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
        0.5,
        color,
        thickness,
    )


def draw_center(frame, box, color=(0, 255, 0), thickness=1):
    x1, y1, x2, y2 = box
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    draw_dot(frame, center, color, thickness)
    return center
