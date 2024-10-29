import asyncio
from bleak import BleakClient, BleakScanner


# 获取设备地址
async def get_device_address():
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"Device: {device.name}, Address: {device.address}")


async def get_device_service(address):
    async with BleakClient(address) as client:
        print(f"Connected: {client.is_connected}")
        services = await client.get_services()
        for service in services:
            print(f"Service UUID: {service.uuid}")
            for characteristic in service.characteristics:
                print(f"  Characteristic UUID: {characteristic.uuid}")


# asyncio.run(get_device_address())



