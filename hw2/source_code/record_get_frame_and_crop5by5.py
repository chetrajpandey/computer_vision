#!/usr/bin/env python3

import cv2
import depthai as dai
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1200, 900)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)
i = 0
# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    result = cv2.VideoWriter('10sec_vid.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, (1200, 900))
    # The duration in seconds of the video captured
    #Using 7 because of the delay while starting the camera, it captures 10 sec video for me
    capture_duration = 7
    start_time = time.time()
    while( int(time.time() - start_time) < capture_duration ):
        videoIn = video.get()
        result.write(videoIn.getCvFrame())
        frame = videoIn.getCvFrame()
        sky = frame[200:205, 200:205]
        # Get BGR frame from NV12 encoded video frame to show with opencv
        # Visualizing the frame on slower hosts might have overhead
        cv2.imshow("video", videoIn.getCvFrame())
        cv2.imwrite('frames/Frame'+str(i)+'.jpg', videoIn.getCvFrame())
        cv2.imwrite('crop/Frame'+str(i)+'.jpg', sky)
        i+=1

        if cv2.waitKey(1) == ord('q'):
            break

#video.release()
result.release()
	
# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
