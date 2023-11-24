import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

cap = cv2.VideoCapture(0)

detector_maos = mp.solutions.hands.Hands(max_num_hands=1)

classes = ['A', 'B', 'C', 'D', 'E']
modelo = load_model('keras_model.h5')
dados = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    sucesso, img = cap.read()
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = detector_maos.process(frame_rgb)
    pontos_maos = resultados.multi_hand_landmarks
    altura, largura, _ = img.shape

    if pontos_maos is not None:
        for mao in pontos_maos:
            x_max = 0
            y_max = 0
            x_min = largura
            y_min = altura
            for lm in mao.landmark:
                x, y = int(lm.x * largura), int(lm.y * altura)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(img, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)

            try:
                img_corpo = img[y_min-50:y_max+50, x_min-50:x_max+50]
                img_corpo = cv2.resize(img_corpo, (224, 224))
                img_array = np.asarray(img_corpo)
                imagem_array = (img_array.astype(np.float32) / 127.0) - 1
                dados[0] = imagem_array
                previsao = modelo.predict(dados)
                indice_val = np.argmax(previsao)
                #print(classes[indice_val])
                cv2.putText(img, classes[indice_val], (x_min-50, y_min-65), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)

            except:
                continue

    cv2.imshow('Imagem', img)
    cv2.waitKey(1)
