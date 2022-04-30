import cv2 
import depthai as dai

#create a pipeline
pipeline = dai.Pipeline()

#define the source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")

#properties

camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1920, 1080)
camRgb.setVideoSize(1280, 720)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

#Linking
camRgb.video.link(xoutVideo.input)

with dai.Device(pipeline) as device:

    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    img_count=0
    frames = []
    while True:
        videoIn = video.get()
        output = videoIn.getCvFrame()
        #Get BGR from NV12 encoded video frame to show with opencv
        cv2.imshow("video", output)

        
        if cv2.waitKey(1) == ord('c'):
            frames.append(output)
            img_count +=1
            print(img_count, 'Image Captured')
            
        if cv2.waitKey(1) == ord('p'):
            print('Creating Panaroma...')
            if img_count < 2:
                print('Pictures not enough! Move Camera and Press c')
            else:
                stitcher=cv2.Stitcher.create()
                res_code, panaroma =stitcher.stitch(frames)
                if res_code != cv2.STITCHER_OK:
                    print("Stitching Failed!")
                else:
                    print('Showing panaroma..')
                    cv2.imshow('Panaroma', panaroma)
                    cv2.imwrite('panaroma.jpg', panaroma)
                    print('Your Panorama is Saved!!!')
        
        if cv2.waitKey(1) == ord('q'):
            break
