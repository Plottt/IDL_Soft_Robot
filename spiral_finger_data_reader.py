#!/usr/bin/env python3

import serial
import time

def main():
    serial_port = 'COM4'     # For Windows, often "COM3" or "COM4"
    # serial_port = '/dev/ttyUSB0'  # For Linux
    # serial_port = '/dev/tty.usbserial'  # For macOS

    baud_rate = 2000000         # Change if your Arduino is using a different baud rate
    output_file = 'IDL_Data_Collection/StuffedAnimal3.txt'

    # Open a connection to the serial port
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print(f"Connected to {serial_port} at {baud_rate} baud.")
    except serial.SerialException as e:
        print(f"Error opening serial port {serial_port}: {e}")
        return

    # Open the output file in append mode
    with open(output_file, 'a') as f:
        print(f"Logging data to {output_file}. Press Ctrl+C to stop.\n")
        try:
            while True:
                # Read one line from the serial buffer
                if ser.in_waiting > 0:
                    line = ser.readline().decode(errors='replace').rstrip('\n')
                    if line:  # Only log non-empty lines
                        # Print to console
                        print(line)

                        # Write the line to file with a timestamp
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{timestamp}, {line}\n")
                        f.flush()  # Ensure data is written to the file immediately
        except KeyboardInterrupt:
            print("\nLogging stopped by user.")
        finally:
            ser.close()

if __name__ == "__main__":
    main()
