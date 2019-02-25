import picamera
import picamera.array
import numpy as np
from time import sleep, time

class PiCam:
    def __init__(self):
        self.camera = picamera.PiCamera()
        self.camera.start_preview()
        # Camera warm-up time
        sleep(2)

    def capture_rgb(self):
        output = picamera.array.PiRGBArray(self.camera)
        self.camera.capture(output, 'rgb')
        return output.array

    def capture_greyscale(self):
        output = picamera.array.PiRGBArray(self.camera)
        self.camera.capture(output, 'rgb')
        return np.divide(
            output.array.reshape(
                (output.array.shape[0] * output.array.shape[1],
                output.array.shape[2]))
            .dot([0.299, 0.587, 0.114]), 255.0)

if __name__ == "__main__":
    cam = PiCam()
    start = time()
    for x in range(5):
        output = cam.capture_greyscale()
        #print(output.shape)
        #print(output.sum() / len(output))
        #print(output)
    print(time() - start)
