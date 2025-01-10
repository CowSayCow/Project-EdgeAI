import time
import cv2
import requests
from ultralytics import YOLO
import json

# YOLOv8モデルの読み込み（適切なモデルパスに置き換えてください）
model = YOLO("yolov8n.pt")  # "yolov8n.pt"の部分はご使用のモデルファイルに合わせて変更してください

# HTTP送信先の設定
url = "ここにIPアドレス"  # IPアドレスとエンドポイントを指定

# # 任意の動画ファイルを指定してキャプチャ開始
# video_path = "video_path.mp4"  # 読み込みたい動画ファイルのパスに置き換え
# cap = cv2.VideoCapture(video_path)

# カメラから映像をキャプチャ
cap = cv2.VideoCapture(0)  # 0は通常デフォルトのカメラを指します

frame_counter = 0
last_send_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # 5フレームに1回だけYOLOの推論を実行
    if frame_counter % 10 == 0:
    
        # YOLOv8で画像を推論
        results = model(frame, verbose=False)
        
        # 検出されたオブジェクトをフィルタリング（人のみ）
        human_data = []
        for result in results[0].boxes:
            if result.cls == 0:  # クラス0は通常「人」を指します
                x1, y1, x2, y2 = result.xyxy[0]
                confidence = float(result.conf)
                human_data.append({
                    "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence
                })

        # 10秒ごとにデータを送信
        current_time = time.time()
        if current_time - last_send_time >= 10:
            if human_data:
                try:
                    # 検出された人数をcurrent_countに設定
                    data = {
                        "id": 1,  # 固定ID
                        "name": "コンビニ前",  # 固定名
                        "max_capacity": 30,  # 固定の最大収容人数
                        "current_count": len(human_data)  # 検出された人数
                    }
                    # データをHTTP POSTリクエストで送信
                    response = requests.post(url, json=data)
                    print("送信完了:", response.status_code)
                except requests.exceptions.RequestException as e:
                    print("送信エラー:", e)

            last_send_time = current_time
            
    frame_counter += 1
        
    # 検出結果を表示
    for person in human_data:
        x1, y1, x2, y2 = person["bounding_box"]
        confidence = person["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #   検出された人数をコンソールに表示
    print(f"検出された人数: {len(human_data)}")
    
    cv2.imshow("YOLOv8 Human Detection", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()

print("終了しました")
