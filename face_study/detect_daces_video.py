# 사용법
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# 인수를 구문 분석하고 인수를 구문 분석합니다.
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="deploy.prototxt.txt")
ap.add_argument("-m", "--model", required=True,
	help="pres10_300x300_ssd_iter_140000.caffemodel")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 디스크에서 직렬화 된 모델로드
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 비디오 스트림을 초기화하고 cammera센서가 워밍업하도록 허용
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 비디오 스트림의 프레임을 반복합니다.
while True:
	# 스레드 된 비디오 스트림에서 프레임을 가져와 크기 조정
	# 최대 너비는 400 픽셀입니다.
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
 
	# 프레임 크기를 가져 와서 blob으로 변환
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# 네트워크를 통해 blob을 전달하고 탐지함
	# 예측 개수
	net.setInput(blob)
	detections = net.forward()

	# 탐지 루프
	for i in range(0, detections.shape[2]):
		# 다음과 관련된 신뢰도 (i.e., probability) 추출
		# prediction
		confidence = detections[0, 0, i, 2]

		# 컨피던스가 최소 신뢰도보다 큼
		# 
		if confidence < args["confidence"]:
			continue

		# bbox의 (x, y) 좌표를 계산합니다.
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# 연관된 얼굴과 함께 얼굴의 경계상자를 그린다.
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()