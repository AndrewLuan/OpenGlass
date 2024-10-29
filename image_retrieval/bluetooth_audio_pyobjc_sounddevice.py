import objc
from Foundation import NSObject, NSRunLoop, NSDate
import sounddevice as sd
from CoreBluetooth import CBCentralManager, CBPeripheral
import numpy as np
import wave
#bluetooth for mac

# 动态获取 CBCentralManagerDelegate 和 CBPeripheralDelegate 协议
CBCentralManagerDelegate = objc.protocolNamed('CBCentralManagerDelegate')
CBPeripheralDelegate = objc.protocolNamed('CBPeripheralDelegate')

class CentralManagerDelegate(NSObject, protocols=[CBCentralManagerDelegate]):
    def initWithTarget_(self, target):
        self = objc.super(CentralManagerDelegate, self).init()
        if self is None:
            return None
        self.target = target
        return self

    def centralManagerDidUpdateState_(self, central):
        self.target.centralManagerDidUpdateState_(central)

    def centralManager_didDiscoverPeripheral_advertisementData_RSSI_(self, central, peripheral, advertisementData, RSSI):
        self.target.centralManager_didDiscoverPeripheral_advertisementData_RSSI_(central, peripheral, advertisementData, RSSI)

    def centralManager_didConnectPeripheral_(self, central, peripheral):
        self.target.centralManager_didConnectPeripheral_(central, peripheral)

    def centralManager_didFailToConnectPeripheral_error_(self, central, peripheral, error):
        self.target.centralManager_didFailToConnectPeripheral_error_(central, peripheral, error)

class PeripheralDelegate(NSObject, protocols=[CBPeripheralDelegate]):
    def initWithTarget_(self, target):
        self = objc.super(PeripheralDelegate, self).init()
        if self is None:
            return None
        self.target = target
        return self

    def peripheral_didDiscoverServices_(self, peripheral, error):
        self.target.peripheral_didDiscoverServices_(peripheral, error)

    def peripheral_didDiscoverCharacteristicsForService_error_(self, peripheral, service, error):
        self.target.peripheral_didDiscoverCharacteristicsForService_error_(peripheral, service, error)

    def peripheral_didUpdateValueForCharacteristic_error_(self, peripheral, characteristic, error):
        self.target.peripheral_didUpdateValueForCharacteristic_error_(peripheral, characteristic, error)

class BluetoothManager(NSObject):
    def init(self):
        self = objc.super(BluetoothManager, self).init()
        if self is None:
            return None

        self.central_manager_delegate = CentralManagerDelegate.alloc().initWithTarget_(self)
        self.peripheral_delegate = PeripheralDelegate.alloc().initWithTarget_(self)
        self.central_manager = CBCentralManager.alloc().initWithDelegate_queue_(self.central_manager_delegate, None)
        self.target_peripheral = None
        return self

    def centralManagerDidUpdateState_(self, central):
        if central.state() == 5:  # Powered On
            print("Bluetooth is powered on. Scanning for devices...")
            self.central_manager.scanForPeripheralsWithServices_options_(None, None)
        else:
            print("Bluetooth is not available.")
    
    def centralManager_didDiscoverPeripheral_advertisementData_RSSI_(self, central, peripheral, advertisementData, RSSI):
        print(f"Discovered: {peripheral.name()} - {peripheral.identifier().UUIDString()}")
        if peripheral.name() == "starry’s AirPods":  # Replace with your device name
            self.target_peripheral = peripheral
            self.central_manager.stopScan()
            self.central_manager.connectPeripheral_options_(peripheral, None)

    def centralManager_didConnectPeripheral_(self, central, peripheral):
        print(f"Connected to {peripheral.name()}")
        peripheral.setDelegate_(self.peripheral_delegate)
        peripheral.discoverServices_(None)
    
    def centralManager_didFailToConnectPeripheral_error_(self, central, peripheral, error):
        print(f"Failed to connect to {peripheral.name()}")

    def peripheral_didDiscoverServices_(self, peripheral, error):
        for service in peripheral.services():
            print(f"Discovered service: {service.UUID()}")
            peripheral.discoverCharacteristics_forService_(None, service)

    def peripheral_didDiscoverCharacteristicsForService_error_(self, peripheral, service, error):
        for characteristic in service.characteristics():
            print(f"Discovered characteristic: {characteristic.UUID()}")

    def peripheral_didUpdateValueForCharacteristic_error_(self, peripheral, characteristic, error):
        print(f"Received data from {characteristic.UUID()}: {characteristic.value()}")

def start_bluetooth_scan():
    manager = BluetoothManager.alloc().init()
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(10))
    return manager.target_peripheral is not None

def record_audio():
    # 设置音频参数
    CHUNK = 1024  # 每次读取的帧数
    FORMAT = 'int16'  # 16 位整数格式
    CHANNELS = 1  # 单声道
    RATE = 44100  # 采样率 44.1kHz

    # 查找蓝牙耳机麦克风设备索引
    device_index = None
    for device in sd.query_devices():
        if "starry’s AirPods" in device['name'].lower() and device['max_input_channels'] > 0:
            device_index = device['index']
            break

  #  if device_index is None:
  #      raise Exception("bluetooth earphone microphone not found")

    print("* recording")

    # 录制音频数据
    frames = []
    def callback(indata, frames_count, time, status):
        if status:
            print(status)
        frames.append(indata.copy())

    with sd.InputStream(samplerate=RATE, device=device_index, channels=CHANNELS, dtype=FORMAT, callback=callback):
        sd.sleep(int(10000))  # 录制 5 秒钟

    print("* done recording")

    # 保存 WAV 文件
    wf = wave.open("recording.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(np.dtype(np.int16).itemsize)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    if start_bluetooth_scan():
        print("Bluetooth device connected, starting audio recording...")
        record_audio()
    else:
        print("Failed to find and connect to Bluetooth device.")