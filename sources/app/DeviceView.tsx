//import * as React from 'react';
//import { ActivityIndicator, Image, ScrollView, Text, TextInput, View, TouchableOpacity } from 'react-native';
//import { toBase64Image } from '../utils/base64';
//import { Agent } from '../agent/Agent';
//import { InvalidateSync } from '../utils/invalidateSync';
//import { textToSpeech } from '../modules/openai';
//import { rotateImage } from '../modules/imaging';
//import JSZip from 'jszip';
//import { saveAs } from 'file-saver';  // You'll need to install file-saver as well: npm install file-saver
//
//
//function usePhotos(device: BluetoothRemoteGATTServer) {
//    const [photos, setPhotos] = React.useState<{ data: Uint8Array, url: string }[]>([]);
//    const [subscribed, setSubscribed] = React.useState<boolean>(false);
//
//    React.useEffect(() => {
//        if (!device) {
//            console.error('No device provided');
//            return;
//        }
//
//        let previousChunk = -1;
//        let buffer: Uint8Array = new Uint8Array(0);
//
//        const onChunk = (id: number | null, data: Uint8Array) => {
//            if (previousChunk === -1) {
//                if (id === null) {
//                    return;
//                } else if (id === 0) {
//                    previousChunk = 0;
//                    buffer = new Uint8Array(0);
//                } else {
//                    return;
//                }
//            } else {
//                if (id === null) {
//                    console.log('Photo received', buffer);
//                    rotateImage(buffer, '270').then((rotated) => {
//                        console.log('Rotated photo', rotated);
//                        const blob = new Blob([rotated], { type: 'image/jpeg' });
//                        const url = URL.createObjectURL(blob);
//                        setPhotos((p) => [...p, { data: rotated, url }]);
//                    });
//                    previousChunk = -1;
//                    return;
//                } else {
//                    if (id !== previousChunk + 1) {
//                        previousChunk = -1;
//                        console.error('Invalid chunk', id, previousChunk);
//                        return;
//                    }
//                    previousChunk = id;
//                }
//            }
//            buffer = new Uint8Array([...buffer, ...data]);
//        };
//
//        const startNotifications = async () => {
//            try {
//                const service = await device.getPrimaryService('19B10000-E8F2-537E-4F6C-D104768A1214'.toLowerCase());
//                const photoCharacteristic = await service.getCharacteristic('19b10005-e8f2-537e-4f6c-d104768a1214');
//                await photoCharacteristic.startNotifications();
//                setSubscribed(true);
//
//                const handleCharacteristicValueChanged = (e: Event) => {
//                    const target = e.target as BluetoothRemoteGATTCharacteristic;
//                    const value = target.value!;
//                    const array = new Uint8Array(value.buffer);
//
//                    if (array[0] === 0xff && array[1] === 0xff) {
//                        onChunk(null, new Uint8Array());
//                    } else {
//                        const packetId = array[0] + (array[1] << 8);
//                        const packet = array.slice(2);
//                        onChunk(packetId, packet);
//                    }
//                };
//
//                photoCharacteristic.addEventListener('characteristicvaluechanged', handleCharacteristicValueChanged);
//
//                return () => {
//                    photoCharacteristic.removeEventListener('characteristicvaluechanged', handleCharacteristicValueChanged);
//                    photoCharacteristic.stopNotifications().catch(console.error);
//                };
//            } catch (error) {
//                console.error('Failed to start notifications', error);
//            }
//        };
//
//        startNotifications();
//    }, [device]);
//
//    return [subscribed, photos] as const;
//}
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//export const DeviceView = React.memo((props: { device: BluetoothRemoteGATTServer }) => {
//    const [subscribed, photos] = usePhotos(props.device);
//    const agent = React.useMemo(() => new Agent(), []);
//    const agentState = agent.use();
//
//    const processedPhotos = React.useRef<Uint8Array[]>([]);
//    const sync = React.useMemo(() => {
//        let processed = 0;
//        return new InvalidateSync(async () => {
//            if (processedPhotos.current.length > processed) {
//                let unprocessed = processedPhotos.current.slice(processed);
//                processed = processedPhotos.current.length;
//                await agent.addPhoto(unprocessed);
//            }
//        });
//    }, []);
//    React.useEffect(() => {
//        processedPhotos.current = photos.map(photo => photo.data);
//        sync.invalidate();
//    }, [photos]);
//
//    React.useEffect(() => {
//        if (agentState.answer) {
//            textToSpeech(agentState.answer);
//        }
//    }, [agentState.answer]);
//
//    const downloadAllPhotos = async () => {
//        const zip = new JSZip();
//        photos.forEach((photo, index) => {
//            zip.file(`photo_${index}.jpg`, photo.data);
//        });
//        const content = await zip.generateAsync({ type: 'blob' });
//        saveAs(content, 'photos.zip');
//    };
//
//    return (
//        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
//            <View style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
//            <TouchableOpacity style={{ backgroundColor: 'blue', padding: 10, borderRadius: 5, marginTop: 20 }} onPress={downloadAllPhotos}>
//                    <Text style={{ color: 'white' }}>Download All Photos</Text>
//                </TouchableOpacity>
//                <ScrollView style={{ flex: 1 }}>
//                    <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
//                        {photos.map((photo, index) => (
//                            <View key={index} style={{ margin: 10 }}>
//                                <Image style={{ width: 100, height: 100 }} source={{ uri: toBase64Image(photo.data) }} />
//                                <TouchableOpacity onPress={() => {
//                                    const link = document.createElement('a');
//                                    link.href = photo.url;
//                                    link.download = `photo_${index}.jpg`;
//                                    document.body.appendChild(link);
//                                    link.click();
//                                    document.body.removeChild(link);
//                                }}>
//                                    <Text style={{ color: 'blue', textAlign: 'center' }}>Download</Text>
//                                </TouchableOpacity>
//                            </View>
//                        ))}
//                    </View>
//                </ScrollView>                
//            </View>
//
//            <View style={{ backgroundColor: 'rgb(28 28 28)', height: 600, width: 600, borderRadius: 64, flexDirection: 'column', padding: 64 }}>
//                <View style={{ flexGrow: 1, justifyContent: 'center', alignItems: 'center' }}>
//                    {agentState.loading && (<ActivityIndicator size="large" color={"white"} />)}
//                    {agentState.answer && !agentState.loading && (<ScrollView style={{ flexGrow: 1, flexBasis: 0 }}><Text style={{ color: 'white', fontSize: 32 }}>{agentState.answer}</Text></ScrollView>)}
//                </View>
//                <TextInput
//                    style={{ color: 'white', height: 64, fontSize: 32, borderRadius: 16, backgroundColor: 'rgb(48 48 48)', padding: 16 }}
//                    placeholder='How can I help you?'
//                    placeholderTextColor={'#888'}
//                    readOnly={agentState.loading}
//                    onSubmitEditing={(e) => agent.answer(e.nativeEvent.text)}
//                />
//            </View>
//        </View>
//    );
//});


import * as React from 'react';
import { ActivityIndicator, Image, ScrollView, Text, TextInput, View, TouchableOpacity, PanResponder, Animated, StyleSheet } from 'react-native';
import { toBase64Image } from '../utils/base64';
import { Agent } from '../agent/Agent';
import { InvalidateSync } from '../utils/invalidateSync';
import { textToSpeech } from '../modules/openai';
import { rotateImage } from '../modules/imaging';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';  // You'll need to install file-saver as well: npm install file-saver

function usePhotos(device: BluetoothRemoteGATTServer) {
    const [photos, setPhotos] = React.useState<{ data: Uint8Array, url: string }[]>([]);
    const [subscribed, setSubscribed] = React.useState<boolean>(false);

    React.useEffect(() => {
        if (!device) {
            console.error('No device provided');
            return;
        }

        let previousChunk = -1;
        let buffer: Uint8Array = new Uint8Array(0);

        const onChunk = (id: number | null, data: Uint8Array) => {
            if (previousChunk === -1) {
                if (id === null) {
                    return;
                } else if (id === 0) {
                    previousChunk = 0;
                    buffer = new Uint8Array(0);
                } else {
                    return;
                }
            } else {
                if (id === null) {
                    console.log('Photo received', buffer);
                    rotateImage(buffer, '270').then((rotated) => {
                        console.log('Rotated photo', rotated);
                        const blob = new Blob([rotated], { type: 'image/jpeg' });
                        const url = URL.createObjectURL(blob);
                        setPhotos((p) => [...p, { data: rotated, url }]);
                    });
                    previousChunk = -1;
                    return;
                } else {
                    if (id !== previousChunk + 1) {
                        previousChunk = -1;
                        console.error('Invalid chunk', id, previousChunk);
                        return;
                    }
                    previousChunk = id;
                }
            }
            buffer = new Uint8Array([...buffer, ...data]);
        };

        const startNotifications = async () => {
            try {
                const service = await device.getPrimaryService('19B10000-E8F2-537E-4F6C-D104768A1214'.toLowerCase());
                const photoCharacteristic = await service.getCharacteristic('19b10005-e8f2-537e-4f6c-d104768a1214');
                await photoCharacteristic.startNotifications();
                setSubscribed(true);

                const handleCharacteristicValueChanged = (e: Event) => {
                    const target = e.target as BluetoothRemoteGATTCharacteristic;
                    const value = target.value!;
                    const array = new Uint8Array(value.buffer);

                    if (array[0] === 0xff && array[1] === 0xff) {
                        onChunk(null, new Uint8Array());
                    } else {
                        const packetId = array[0] + (array[1] << 8);
                        const packet = array.slice(2);
                        onChunk(packetId, packet);
                    }
                };

                photoCharacteristic.addEventListener('characteristicvaluechanged', handleCharacteristicValueChanged);

                return () => {
                    photoCharacteristic.removeEventListener('characteristicvaluechanged', handleCharacteristicValueChanged);
                    photoCharacteristic.stopNotifications().catch(console.error);
                };
            } catch (error) {
                console.error('Failed to start notifications', error);
            }
        };

        startNotifications();
    }, [device]);

    return [subscribed, photos] as const;
}

export const DeviceView = React.memo((props: { device: BluetoothRemoteGATTServer }) => {
    const [subscribed, photos] = usePhotos(props.device);
    const agent = React.useMemo(() => new Agent(), []);
    const agentState = agent.use();

    const [dialogSize, setDialogSize] = React.useState({ width: 600, height: 600 });
    const animatedWidth = React.useRef(new Animated.Value(dialogSize.width)).current;
    const animatedHeight = React.useRef(new Animated.Value(dialogSize.height)).current;

    const processedPhotos = React.useRef<Uint8Array[]>([]);
    const sync = React.useMemo(() => {
        let processed = 0;
        return new InvalidateSync(async () => {
            if (processedPhotos.current.length > processed) {
                let unprocessed = processedPhotos.current.slice(processed);
                processed = processedPhotos.current.length;
                await agent.addPhoto(unprocessed);
            }
        });
    }, []);
    React.useEffect(() => {
        processedPhotos.current = photos.map(photo => photo.data);
        sync.invalidate();
    }, [photos]);

    React.useEffect(() => {
        if (agentState.answer) {
            textToSpeech(agentState.answer);
        }
    }, [agentState.answer]);

    const downloadAllPhotos = async () => {
        const zip = new JSZip();
        photos.forEach((photo, index) => {
            zip.file(`photo_${index}.jpg`, photo.data);
        });
        const content = await zip.generateAsync({ type: 'blob' });
        saveAs(content, 'photos.zip');
    };

    const panResponder = React.useMemo(() => PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onPanResponderMove: (event, gestureState) => {
            const newWidth = Math.max(100, dialogSize.width + gestureState.dx);
            const newHeight = Math.max(100, dialogSize.height + gestureState.dy);

            animatedWidth.setValue(newWidth);
            animatedHeight.setValue(newHeight);
            setDialogSize({ width: newWidth, height: newHeight });
        },
    }), [dialogSize]);

    return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <View style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <TouchableOpacity style={{ backgroundColor: 'blue', padding: 10, borderRadius: 5, marginTop: 20 }} onPress={downloadAllPhotos}>
                    <Text style={{ color: 'white' }}>Download All Photos</Text>
                </TouchableOpacity>
                <ScrollView style={{ flex: 1 }}>
                    <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
                        {photos.map((photo, index) => (
                            <View key={index} style={{ margin: 10 }}>
                                <Image style={{ width: 100, height: 100 }} source={{ uri: toBase64Image(photo.data) }} />
                                <TouchableOpacity onPress={() => {
                                    const link = document.createElement('a');
                                    link.href = photo.url;
                                    link.download = `photo_${index}.jpg`;
                                    document.body.appendChild(link);
                                    link.click();
                                    document.body.removeChild(link);
                                }}>
                                    <Text style={{ color: 'blue', textAlign: 'center' }}>Download</Text>
                                </TouchableOpacity>
                            </View>
                        ))}
                    </View>
                </ScrollView>
            </View>

            <Animated.View
                style={[
                    styles.dialogContainer,
                    {
                        width: animatedWidth,
                        height: animatedHeight,
                    },
                ]}
                {...panResponder.panHandlers}
            >
                <View style={styles.dialogContent}>
                    <View style={{ flexGrow: 1, justifyContent: 'center', alignItems: 'center' }}>
                        {agentState.loading && (<ActivityIndicator size="large" color={"white"} />)}
                        {agentState.answer && !agentState.loading && (<ScrollView style={{ flexGrow: 1, flexBasis: 0 }}><Text style={{ color: 'white', fontSize: 32 }}>{agentState.answer}</Text></ScrollView>)}
                    </View>
                    <TextInput
                        style={{ color: 'white', height: 64, fontSize: 32, borderRadius: 16, backgroundColor: 'rgb(48 48 48)', padding: 16 }}
                        placeholder='How can I help you?'
                        placeholderTextColor={'#888'}
                        readOnly={agentState.loading}
                        onSubmitEditing={(e) => agent.answer(e.nativeEvent.text)}
                    />
                </View>
                <View style={styles.resizeHandle} />
            </Animated.View>
        </View>
    );
});

const styles = StyleSheet.create({
    dialogContainer: {
        backgroundColor: 'rgb(28, 28, 28)',
        borderRadius: 16,
        position: 'absolute',
        padding: 16,
    },
    dialogContent: {
        flex: 1,
    },
    resizeHandle: {
        width: 20,
        height: 20,
        backgroundColor: 'gray',
        position: 'absolute',
        bottom: 0,
        right: 0,
        borderRadius: 10,
    },
});