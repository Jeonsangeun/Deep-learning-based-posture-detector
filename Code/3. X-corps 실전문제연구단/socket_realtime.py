import socket
import numpy as np
from keras.models import load_model
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime

HOST = '210.125.112.77'
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("대기중입니다")

client_socket, addr = server_socket.accept()

print('Connected by', addr)
print(str(addr), "에 접속되었습니다.")

model = load_model('pos_model_3')
cred = credentials.Certificate('test1-3a331-57a0f8f0e5fe.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
result_chn = 0
real_pos = 0
count = 0
def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

while True:
    data = recvall(client_socket, 3840)
    if not data:
        break


    aaa = np.frombuffer(data, dtype=np.float64)
    aaa = np.reshape(aaa, (20*8*3))
    aaa = np.expand_dims(aaa, axis=0)*1
    result = np.argmax(model.predict(aaa))

    if result_chn == int(result):
        count += 1
    else:
        result_chn = int(result)
        count = 0

    if count >= 2:
        real_pos = int(result_chn)

    print("result : ", result+1)
    print("현재", int(real_pos)+1, "번 자세입니다")
    print("=====================================================================")
    dt = datetime.datetime.now()
    month = dt.month
    day = dt.day
    if month < 10:
        month = '0' + str(month)
    if day < 10:
        day = '0' + str(day)
    realtime_posture = db.collection(str(month)).document(str(day))  # 자세 콜렉션
    realtime_posture.set({str(dt.hour) + '/' + str(dt.minute) + '/' + str(dt.second): int(real_pos)+ 1}, merge=True)

    realtime_posture = db.collection('user').document('position')  # 자세 콜렉션
    realtime_posture.set({'now' : int(real_pos)+ 1})


# 소켓을 닫습니다.
client_socket.close()
server_socket.close()