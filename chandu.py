import serial 
import time
while(1):
	a = []
	b=[]
	ser = serial.Serial('/dev/cu.usbmodem1411', 9600)
	file_object = open("serial.txt", "r")
	a = file_object.readline().split()
	# print a
	b = file_object.readline().split()
	if(len(b)>0):
		if a[0] == '0' and b[0]== '0' :
			print a
			print b
			ser.write('a')
		if a[0] == '1' and b[0]== '0' :
			print a
			print b
			ser.write('b')
		if a[0] == '0' and b[0] == '1' :
			print a
			print b
			ser.write('c')
		if a[0] == '1' and b[0] == '1' :
			print a
			print b
			ser.write('d')	
		
	#if(len(b)!=0):
	#	a.append(b[0])
	#a.append('\n')
	#if(len(a) ==2):
	#	ser.write(a) 
	#print "Going"
	#print a
	time.sleep(.200)