import socket

#remote host name and port
HOST='pi1'
PORT=8888

ADDR=(HOST,PORT)
BUF_SIZE = 1024

client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
client.connect(ADDR)
client.send('client say:Here comes an object from My Mac!')
data = client.recv(BUF_SIZE)
if data:
    print 'client recv from', HOST,'----',data
client.close()