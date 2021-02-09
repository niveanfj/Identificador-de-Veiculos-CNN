import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import resnet50
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16

carro = ["streetcar", "tractor", "ambulance", "tow_truck", "recreational_vehicle", "moving_van", "passenger_car",
         "Model_T", "trolleybus", "police_van", "limousine", "sports_car", "pickup", "school_bus", "garbage_truck",
         "minibus", "minivan"]
model = resnet50.ResNet50(weights='imagenet')
#model = VGG16(weights='imagenet')    # Atribui ao modelo os pesos da imagenet

# Entrada do video
video = cv2.VideoCapture('videoeditado.mp4')

# Saida de video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
saida = cv2.VideoWriter('resultado.avi', fourcc, 30.0, (int(video.get(3)), (int(video.get(4)))))

while True:
    # Area de interesse
    ret, frame = video.read()
    if ret:
        cv2.rectangle(frame, (640, 380), (770, 470), (128, 128, 128), 2)    # desenha o retangulo da area de interesse
        roi = frame[380:470, 640:770]     # dimensão da area de interesse

        # Imagem pra PIL
        a_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        a_roi = cv2.resize(a_roi, (224, 224))

        a_roi = img_to_array(a_roi)
        a_roi = a_roi.reshape((1, a_roi.shape[0], a_roi.shape[1], a_roi.shape[2]))
        a_roi = preprocess_input(a_roi)
        rec = model.predict(a_roi)

        label = decode_predictions(rec)
        label = label[0][0]

        # Texto da condição de detecção
        if label[1] in carro:
            cv2.putText(frame, "Carro", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)

    saida.write(frame)           # Grava o video de saida (por frame)
    cv2.imshow('frame', frame)   # Mostra o processo

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
saida.release()
cv2.destroyAllWindows()
