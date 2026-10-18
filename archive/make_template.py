import cv2
from pathlib import Path

cap = cv2.VideoCapture("videos/sample.mp4")
frame_no = int(input("Frame number with a clear tag: "))
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
ok, frame = cap.read()
assert ok, "Could not read that frame"

# Drag a tight box around the tag, press ENTER to confirm
x, y, w, h = cv2.selectROI("Select the tag", frame, showCrosshair=False)
cv2.destroyAllWindows()

Path("templates").mkdir(exist_ok=True)
cv2.imwrite("templates/tag.png", frame[y:y+h, x:x+w])
print(f"Saved template: {w}x{h} px")