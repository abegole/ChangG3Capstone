import serial
import time

# Configure the serial port
ser = serial.Serial('/dev/ttyUSB0', 9600)  # Replace with your actual port and baud rate

try:
    while True:
        # Send data
        ser.write(b'Hello, Serial!\n')
        print("Sent: Hello, Serial!")

        # Receive data
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            print(f"Received: {data}")

        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()