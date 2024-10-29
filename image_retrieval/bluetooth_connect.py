import asyncio
from bleak import BleakScanner
from bleak import BleakClient

# 获取设备地址
async def run():
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"Device: {device.name}, Address: {device.address}")

asyncio.run(run())


# 获取服务和特征UUID
# async def run(address):
#     async with BleakClient(address) as client:
#         print(f"Connected: {client.is_connected}")
#         services = await client.get_services()
#         for service in services:
#             print(f"Service UUID: {service.uuid}")
#             for characteristic in service.characteristics:
#                 print(f"  Characteristic UUID: {characteristic.uuid}")

# # Replace 'DEVICE_ADDRESS' with your device address
# asyncio.run(run("24:58:7C:E3:6F:ED"))

