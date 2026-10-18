import cv2, sys

cap = cv2.VideoCapture("videos/sample.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_no = int(sys.argv[1]) if len(sys.argv) > 1 else 0

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    if not ok:
        break
    cv2.putText(frame, f"frame {frame_no}/{total-1}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("viewer", frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("d"):   frame_no = min(frame_no + 1, total - 1)    # next frame
    elif key == ord("a"): frame_no = max(frame_no - 1, 0)            # previous frame
    elif key == ord("w"): frame_no = min(frame_no + 30, total - 1)   # +30 frames
    elif key == ord("s"): frame_no = max(frame_no - 30, 0)           # -30 frames
    elif key == ord("q"): break

cv2.destroyAllWindows()