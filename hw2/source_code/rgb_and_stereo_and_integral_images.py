import cv2
import depthai as dai
import numpy as np

#Get the frame and return the opencv frame
def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()

#Mono Camera Configuration
def getMonoCamera(pipeline, isLeft):
    mono = pipeline.createMonoCamera()
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    if isLeft:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono

if __name__ == '__main__':
    pipeline = dai.Pipeline()

    #Setting up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft = True)
    monoRight = getMonoCamera(pipeline, isLeft = False)
    camRgb = pipeline.create(dai.node.ColorCamera)

    #Setting RGB Cam Resolution
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setVideoSize(1920, 1080)
    camRgb.setVideoSize(720, 640)
    
    #Setting output Xlink for left camera
    xoutLeft = pipeline.createXLinkOut()
    xoutLeft.setStreamName("left")

    #Setting output XLink for crgb camera
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")

    #Set output Xlink for right camera
    xoutRight = pipeline.createXLinkOut()
    xoutRight.setStreamName("right")

    #Attach cameras to output XLink 
    monoLeft.out.link(xoutLeft.input)
    monoRight.out.link(xoutRight.input)
    camRgb.video.link(xoutRgb.input)
    xoutRgb.input.setBlocking(False)
    xoutRgb.input.setQueueSize(1)
    
    #Connect to Device
    with dai.Device(pipeline) as device:

        #get the output queues.
        leftQueue = device.getOutputQueue(name = 'left', maxSize=1)
        rightQueue = device.getOutputQueue(name = 'right', maxSize = 1)
        rgbQueue = device.getOutputQueue(name="rgb", maxSize=1)
        
        while True:
            #get left Frame
            leftFrame = getFrame(leftQueue)
            
            #get right frame
            rightFrame = getFrame(rightQueue)
            
            #Pair left right camera together
            imOut = np.hstack((leftFrame, rightFrame))

            cv2.imshow("Stereo Pair Feed", imOut)

            rgb = getFrame(rgbQueue)
            rgb = np.array(rgb)
            integral_feed = np.cumsum(rgb, axis=1).cumsum(axis=0)
            integral_feed = cv2.normalize(integral_feed, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)

            cv2.imshow("RGB Feed", rgb)
            cv2.imshow("Integral Image Feed", integral_feed)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break # quit when q is pressed
