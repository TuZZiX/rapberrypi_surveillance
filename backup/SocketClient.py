import socket

#remote host name and port
HOST='pi1'
PORT=8080

ADDR=(HOST,PORT)
BUF_SIZE = 1024

nID = ''
conn = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
conn.connect(ADDR)

choice = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
choice.connect(ADDR)

while 1:
    data = conn.recv(BUF_SIZE)
    if not data:
        break
    print 'Client:',data
    conn.send('Alarm recived!')

    '''nID = raw_input("1 OR 0?")
    if (len(nID) == len('1') or len(nID) == len('0') ):
        print 'Your choice is: %s' % (nID)
        choice.send(nID)
    else:
        print 'error'''
    choice.close()
conn.close()
