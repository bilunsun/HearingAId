import subprocess
import time

# i2c wired to bus 1, on pins 3 - SDA and 5 - SCL
# bluetooth module is configured with address 0x55

def send_class( class_val ):
    # Spawn a process to transfer data over i2c
    command_args = [ "i2ctransfer", "-y", "1", "w1@0x55", str(class_val) ]
    completed_process = subprocess.run(command_args)
    if completed_process.returncode != 0:
        raise ChildProcessError("Failed to send audio class over i2c. Got return code: " + str(completed_process.returncode))


# Kwik test
if __name__ == "__main__":
    for i in range( 0, 4 ):
        send_class(0)
        time.sleep(0.5)
        send_class(1)
        time.sleep(0.5)
        send_class(2)
        time.sleep(0.5)
