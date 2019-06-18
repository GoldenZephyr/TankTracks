import cv2
import numpy as np
import PyCapture2

class VideoSource:

    def __init__(self):
        pass
    

# Generic class for Point Grey Camera using PyCapture2 API
class PGCamera(VideoSource):

    def __init__(self,camera_id,disp_width=831,disp_height=685,raw_width=2464,raw_height=2056):
        try:
            self.bus = PyCapture2.BusManager()
            self.num_cams = self.bus.getNumOfCameras()
            self.camera = PyCapture2.Camera()
            self.uid = self.bus.getCameraFromIndex(camera_id)
            self.camera.connect(self.uid)
            self.print_build_info()
            self.print_camera_info()
            self.disp_width = disp_width
            self.disp_height = disp_height
            self.raw_width = raw_width
            self.raw_height = raw_height
        except Exception as e:
            print(e)
            print ('Could not load camera, will now exit.')
            exit()
            
    def get_frame(self):
    
        image = self.camera.retrieveBuffer()
        row_bytes = float(len(image.getData())) / float(image.getRows());
        new_frame = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols()) );
        new_frame = cv2.resize(new_frame,(int(self.disp_width),int(self.disp_height)))
        #new_frame = cv2.cvtColor(new_frame, cv2.COLOR_GRAY2BGR)
    
        return new_frame
    
    def start_capture(self):
        self.camera.startCapture()
        
    def stop_capture(self):
        self.camera.stopCapture()
    
    def print_build_info(self):
        libVer = PyCapture2.getLibraryVersion()
        print('FlyCapture2 library version: %d %d %d %d' % (libVer[0], libVer[1], libVer[2], libVer[3]))

    def print_camera_info(self):
        cam_info = self.camera.getCameraInfo()
        print('\n*** CAMERA INFORMATION ***\n')
        print('Serial number - %d' % cam_info.serialNumber)
        print('Camera model - %s' % cam_info.modelName)
        print('Camera vendor - %s' % cam_info.vendorName)
        print('Sensor - %s' % cam_info.sensorInfo)
        print('Resolution - %s' % cam_info.sensorResolution)
        print('Firmware version - %s' % cam_info.firmwareVersion)
        print('Firmware build time - %s' % cam_info.firmwareBuildTime)
    
    
#class VideoFile(VideoSource):
    
        