import socket

myname = socket.getfqdn(socket.gethostname())
myaddr = socket.gethostbyname(myname)

HOST=myaddr#'172.20.35.86'
PORT=8888
ADDR=(HOST,PORT)
BUF_SIZE=1024

print("Machine address is %s" %HOST)
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind(ADDR)
server.listen(1)

pi1,addr = server.accept()
print 'server connected by:',addr


#pi1 = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#pi1.connect(ADDR)
pi1.send('Here comes an object!')
data = pi1.recv(BUF_SIZE)
if data:
    print 'recv----',data
pi1.close()

conn,addr = server.accept()
while 1:
    data2 = conn.recv(BUF_SIZE)
    if not data2:
        print 'Error'
    else:
        print 'Decision:',data2
        conn.send('Decision recived!')
        break

conn.close()