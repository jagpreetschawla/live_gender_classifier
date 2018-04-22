import cv2
from multiprocessing import Process, Queue

from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import classification_pb2
from grpc.beta import implementations


face_cascade = cv2.CascadeClassifier('./models/opencv/haarcascade_frontalface_default.xml')

def get_prediction(img):
    channel = implementations.insecure_channel("localhost", 9000)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = classification_pb2.ClassificationRequest()
    request.model_spec.name = "default"
    example = request.input.example_list.examples.add()
    example.features.feature['x'].float_list.value.extend(img.flatten())

    result = stub.Classify(request, 5.0).result.classifications[0]  # 5 secs timeout

    return "male" if result.classes[0].score < result.classes[1].score else "female"

def prediction_thread(inp_q, pred_q):
    while True:
        inp = inp_q.get()
        if inp is None:
            return
        img, faces = inp
        predictions = []
        for i in faces:
            inp = cv2.resize(img[i[0][1]:i[1][1], i[0][0]:i[1][0]], dsize=(25,25))
            predictions.append((i, get_prediction(inp)))
        pred_q.put(predictions)

def get_faces(img_gray):
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    detected_rect = []
    for (x, y, w, h) in faces:
        detected_rect.append(((x,y),(x+w,y+h)))
    return detected_rect

def main():
    cap = cv2.VideoCapture(0)
    inp_q = Queue(2)
    pred_q = Queue(2)
    pred_t = Process(target=prediction_thread, args=(inp_q, pred_q))
    pred_t.start()
    last_pred = []
    pred_q.put(last_pred)
    while True:
        _, img = cap.read()
        img = cv2.resize(img, fx=0.5, fy=0.5, dsize=(0, 0))
        if not pred_q.empty():
            last_pred = pred_q.get()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = get_faces(img)
            inp_q.put((img_gray,faces))
        for i in last_pred:
            pos = i[0]
            pred = i[1]
            cv2.rectangle(img, pos[0], pos[1], color=(255,0,0), thickness=2)
            text_x = (pos[0][0] + pos[1][0]) / 2
            text_y = (pos[0][1] + pos[1][1]) / 2
            cv2.putText(img, pred, (text_x, text_y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (0,0,255), thickness=2)
        cv2.imshow("camera", img)
        if cv2.waitKey(1) == 27:
            break
    inp_q.put(None)
    pred_t.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
