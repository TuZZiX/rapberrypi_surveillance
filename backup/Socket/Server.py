import socket

HOST='pi1'
PORT=8888
ADDR=(HOST,PORT)
BUF_SIZE=1024

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(ADDR)
server.listen(1)

conn,addr = server.accept()
print 'server connected by:',addr
while 1:
    data = conn.recv(BUF_SIZE)
    if not data:
        break
    print 'server recv:',data
    conn.send('Get it!')
conn.close()
