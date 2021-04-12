import socket
import numpy as np
HOST = '210.125.112.77'
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("대기중입니다")

client_socket, addr = server_socket.accept()

print('Connected by', addr)
print(str(addr), "에 접속되었습니다.")
aaa = []
temp_data = []

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

while True:

    data = recvall(client_socket, 4000                                                                                                                                                                                                                                                                                                                                                                  )

    # if not data:
    #     break

    aaa = np.frombuffer(data, dtype=np.float64)
    print(aaa.shape)
    # aaa = np.reshape(aaa, [20, 5, 3])
                                             00
    print('Received from', addr, aaa.shape)

    temp_data.append(aaa)

temp_data = np.array(temp_data)

temp_data = np.delete(temp_data, [0, 1], axis=0)
print(temp_data.shape)
np.save("pos2_5", temp_data)

# 소켓을 닫습니다.
client_socket.close()
server_socket.close()
