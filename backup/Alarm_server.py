import socket

myname = socket.getfqdn(socket.gethostname())
myaddr = socket.gethostbyname(myname)

HOST=myaddr#'172.20.35.86'
PORT=8080
ADDR=(HOST,PORT)
BUF_SIZE=1024

print("Machine address is %s" %HOST)
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(ADDR)
server.listen(1)

conn,addr = server.accept()
print 'server connected by:',addr
while 1:
    data = conn.recv(BUF_SIZE)
    if not data:
        break
    print 'Client:',data
    conn.send('Alarm recived!')
conn.close()
