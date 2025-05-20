import serial
import time
import sys
import os
sys.path.append('..')
from lib.tlvcodec import Encoder, Decoder
import ctypes

# Configure the serial port
ser = serial.Serial('/dev/ttyUSB0', 9600)  # Replace with your actual port and baud rate

# prepare message to send 
# test encoder
encoder = Encoder(199) # device id
encoder.addPacket(987, ctypes.sizeof(ctypes.c_uint32), ctypes.c_uint32(111))
length, buffer = encoder.wrapupBuffer()

def decoder_callback(error, frameHeader, tlvs):
    print(f"Decoder_callback: {error}")
    
    # print out tlvs
    if error.value == 0:
        # print type length value
        for i in range(frameHeader.numTlvs):
            print(f'Type: {tlvs[i][0]}; length: {tlvs[i][1]}')
                
        
                
decoder = Decoder(decoder_callback, True)

try:
    while True:
        # Send data
        ser.write(buffer[:length])
        print(f"Encoder message sent.")
        print(f"Data [{length}]:")	
        for i in range(length):
            print(f'{buffer[i]:02d}', end = ' ')
        print("\n")

        # Receive data
        if ser.in_waiting > 0:
            # data = ser.readline().decode('utf-8').strip()
            # print(f"Received: {data}")
            # the date will not be in the form of string, so we need to read the data as bytes
            data = ser.read(ser.in_waiting)
            decoder.decode(data)

        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
