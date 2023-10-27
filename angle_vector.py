import cv2
import math
import time
from ultralytics import YOLO
from pymycobot.mycobot import MyCobot

model = YOLO('yolov8n-pose.pt')

video_path = 0
capture = cv2.VideoCapture(video_path)
#myCobot接続
mc = MyCobot('/dev/ttyACM0', 115200)


while capture.isOpened():
	success, frame = capture.read()
	if success:
		results = model(frame)
		annotatedFrame = results[0].plot()
		resize = cv2.resize(annotatedFrame,(1400,1050))
		keypoints_tensor = results[0].keypoints.xy
		confidence_score = results[0].keypoints.conf
		if keypoints_tensor is not None and keypoints_tensor.size(1) > 0:
			x_mimi = keypoints_tensor[0][3][0]
			y_mimi = keypoints_tensor[0][3][1]
			conf_mimi = confidence_score[0][3]
			
			x_kosi = keypoints_tensor[0][13][0]
			y_kosi = keypoints_tensor[0][13][1]
			conf_kosi = confidence_score[0][13]
			
			x_kata = keypoints_tensor[0][5][0]
			y_kata = keypoints_tensor[0][5][1]
			conf_kata = confidence_score[0][5]
			
			x_hizi = keypoints_tensor[0][7][0]
			y_hizi = keypoints_tensor[0][7][1]
			conf_hizi = confidence_score[0][7]
			
			x_te = keypoints_tensor[0][9][0]
			y_te = keypoints_tensor[0][9][1]
			conf_te = confidence_score[0][9]
			
			radius = 20  # ブラックアウトする領域の半径（調整可能）
			color = (0, 215, 255)  # ブラックアウトする色（黒色）
			cv2.circle(resize, (int(x_te*2.2), int(y_te*2.2)), radius, color, -1)  # 円を描画してブラックアウト
			
			
			##################################################
			#腰ー肩の水平からの角度を計算
			vector_AB = (x_kata - x_hizi, y_kata - y_hizi)
			# ベクトルABの角度を計算（ラジアン）
			angle_rad1 = math.atan2(vector_AB[1], vector_AB[0])
			# ラジアンから度に変換
			angle_deg1 = math.degrees(angle_rad1)
			mycobot1 = int(angle_deg1)-90
			##################################################
			##################################################
			x1=x_te
			y1=y_te
			
			x2=x_hizi
			y2=y_hizi
			
			x3=x_kata
			y3=y_kata
			
			
			# 3つの点の座標を定義
			point1 = (x1, y1)
			point2 = (x2, y2)
			point3 = (x3, y3)
			# ベクトルを計算
			vector1 = (x2 - x1, y2 - y1)
			vector2 = (x3 - x2, y3 - y2)
			# ベクトルの長さを計算
			length1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
			length2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
			# ドット積を計算
			dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
			# 角度を計算（ラジアン）
			angle_rad = math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0])
			# 角度が右に直角に曲がる場合、0度
			# 角度が左に直角に曲がる場合、180度
			# 直線上の場合、-90度（または+90度、どちらでも構いません）
			if angle_rad > math.pi:
				angle_rad -= 2 * math.pi
			elif angle_rad < -math.pi:
				angle_rad += 2 * math.pi
			# 角度を度数法に変換
			mycobot2 = int(math.degrees(angle_rad))
			##################################################
			if -180 <= mycobot1 and mycobot1 <= 180 and -155 <= mycobot2 and mycobot2 <= 155 and conf_hizi >= 0.75:
				mc.send_angles ([90,-mycobot1,mycobot2,0,-90,0],100)
				mc.set_color(0, 0, 255)
				print("A点の角度（度数法）:", conf_hizi)
			else:
				print("A点の角度（度数法）:", conf_hizi)
				mc.set_color(255, 0, 255)
		else:
			print("キーポイントが存在しないか、データが空です。")
			mc.set_color(0, 0, 0)
		flipped_frame = cv2.flip(resize, 1)
		cv2.imshow("YOLOv8 Inference", flipped_frame)
		



		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	else:
		break

capture.release()
cv2.destroyAllWindows
