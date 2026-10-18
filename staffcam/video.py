import cv2


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def read_frame(cap, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n))
    ok, frame = cap.read()
    return frame if ok else None


def iter_frames(path):
    """Yield (frame_index, frame) for every frame."""
    cap = open_video(path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()
