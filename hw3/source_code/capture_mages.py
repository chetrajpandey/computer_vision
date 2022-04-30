import cv2
import depthai as dai
import numpy as np

def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()

def getMonoCamera(pipeline, isLeft):
    mono = pipeline.createMonoCamera()

    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    if isLeft:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono

def getStereoPair(pipeline, monoLeft, monoRight):
    stereo = pipeline.createStereoDepth()
    stereo.setLeftRightCheck(True)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo


def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y

if __name__=='__main__':

    mouseX = 0
    mouseY = 640
    #Start defining a pipeline
    pipeline = dai.Pipeline()

    #set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft = True)
    monoRight = getMonoCamera(pipeline, isLeft = False)
    #combine left and right cameras to form a stereo pair
    stereo = getStereoPair(pipeline, monoLeft, monoRight)
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setVideoSize(1080, 720)
    #Set XLinkOut for disparity, rectifiedLeft, and rectifiedRight
    xoutDisp = pipeline.createXLinkOut()
    xoutDisp.setStreamName("disparity")

    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")

    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")

    stereo.disparity.link(xoutDisp.input)

    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)

    stereo.rectifiedRight.link(xoutRectifiedRight.input)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("color")
    camRgb.video.link(xoutRgb.input)
    xoutRgb.input.setBlocking(False)
    xoutRgb.input.setQueueSize(1)
    # Pipeline is now defined, now we can connect to the device
    with dai.Device(pipeline, True) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)
        video = device.getOutputQueue(name="color", maxSize=1, blocking=False)

        # Calculate a multiplier for colormapping disparity map
        disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()

        
        
        # Variable use to toggle between side by side view and one frame view.
        sideBySide = False
        i=0

        while True:
            
            # Get disparity map
            disparity = getFrame(disparityQueue)
            videoIn = getFrame(video)

            # Colormap disparity for display
            disparity = (disparity * disparityMultiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
            # Get left and right rectified frame
            leftFrame = getFrame(rectifiedLeftQueue)
            rightFrame = getFrame(rectifiedRightQueue)
            cv2.imshow("Disparity", disparity)
            cv2.imshow("RGB", videoIn)
            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
            if key == ord('c'):
                # Toggle display when t is pressed
                cv2.imwrite('Frame'+str(i)+'.jpg', videoIn)
                i=i+1
