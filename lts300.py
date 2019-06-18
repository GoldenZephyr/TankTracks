""" Uses Kinesis to control Thorlabs long travel stage in XYZ configuration
"""
import clr
import sys
import time
import numpy as np
#import matplotlib.pyplot as plt
import random  # only used for dummy data

from System import String
from System import Decimal
from System.Collections import *

from kbhit import KBHit

# constants
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")

        
# add .net reference and import so python can see .net
clr.AddReference("Thorlabs.MotionControl.Controls")
import Thorlabs.MotionControl.Controls

# print(Thorlabs.MotionControl.Controls.__doc__)

# Add references so Python can see .Net
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")
from Thorlabs.MotionControl.DeviceManagerCLI import *
# from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *

serial_x = '45874644'
serial_y = '45874637'
max_wait = 80000
jog_dist = 1.0

class LTS300:

    def __init__(self,serial_num):
        self.serial_num = serial_num
        self.max_wait = 8000

    
    def jog(self,jog,min=50,max=250,max_wait=0):

        new_pos = float(str(self.device.Position)) + jog
        print(new_pos)

        if new_pos >= min and new_pos <= max:
            
            try:
                # device.Stop(10)
                # if jog > 0:
                    # device.MoveContinuous(Thorlabs.MotionControl.GenericMotorCLI.MotorDirection.Forward)
                # else:
                    # device.MoveContinuous(Thorlabs.MotionControl.GenericMotorCLI.MotorDirection.Backward)
                self.device.SetMoveRelativeDistance(Decimal(jog))
                self.device.MoveRelative(max_wait)
            except Thorlabs.MotionControl.DeviceManagerCLI.DeviceMovingException:
                pass
        


    def initialize(self):
        
        DeviceManagerCLI.BuildDeviceList()
        
        self.device = Thorlabs.MotionControl.IntegratedStepperMotorsCLI.LongTravelStage.CreateLongTravelStage(self.serial_num)
        self.device.ClearDeviceExceptions()
        self.device.Connect(self.serial_num)    
        self.device.WaitForSettingsInitialized(5000)
        self.device.LoadMotorConfiguration(self.serial_num)
        deviceInfo = self.device.GetDeviceInfo()
        self.device.EnableDevice()
        print(deviceInfo.Name, '  ', deviceInfo.SerialNumber)
        






if __name__ == "__main__":

    kb = KBHit()

    device_x = LTS300(serial_x)
    device_y = LTS300(serial_y)
    
    device_x.initialize()
    device_y.initialize()

    #device_list_result = DeviceManagerCLI.BuildDeviceList()
    #print(device_list_result)
    # setup devices
    #device_x = initialize_device(serial_x)
    #device_y = initialize_device(serial_y)

    time.sleep(2.0)

    # home device
    #device_x.Home(max_wait)
    #device_y.Home(max_wait)

    # move to starting position
    device_x.device.MoveTo(Decimal(100), max_wait)
    device_y.device.MoveTo(Decimal(200), max_wait)
  
    time.sleep(1)

    # sweep up and take data
    y_pos = []
    voltage = []
    endpos = 200
    # polling allows us to determin position on the fly
    device_x.device.StartPolling(50)
    device_y.device.StartPolling(50)

    while True:
    
        
    
        if kb.kbhit():
            c = kb.getch()
            
            if ord(c) == 27: # ESC
                break
            print(c)
            
            
            
            if c == 'u':
                print('Up')
                device_y.jog(-jog_dist)
            if c == 'd':
                print('Dn')
                device_y.jog(jog_dist)
            if c == 'l':
                print('Lt')
                device_x.jog(-jog_dist)
            if c == 'r':
                print('Rt')
                device_x.jog(-jog_dist)
            if c == 's':
                device_x.device.StopImmediate()
                device_y.device.StopImmediate()
        
        print(str(device_x.device.Position) + ' : ' + str(device_y.device.Position))
            
        time.sleep(.05)
            
            

    

