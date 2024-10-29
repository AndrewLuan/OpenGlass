import * as React from 'react';
import { ActivityIndicator, Image, ScrollView, Text, TextInput, View, TouchableOpacity} from 'react-native';
import { toBase64Image } from '../utils/base64';
import { Agent } from '../agent/Agent';
import { InvalidateSync } from '../utils/invalidateSync';
import { textToSpeech } from '../modules/openai';
import { rotateImage } from '../modules/imaging';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';  // You'll need to install file-saver as well: npm install file-saver
import { PanGestureHandler, GestureHandlerRootView } from 'react-native-gesture-handler';
import {Animated, PanResponder,} from 'react-native';



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
//    return (
//        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
//            <View style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
//                <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
//                    {photos.map((photo, index) => (
//                        <View key={index} style={{ margin: 10 }}>
//                            <Image style={{ width: 100, height: 100 }} source={{ uri: toBase64Image(photo.data) }} />
//                            <TouchableOpacity onPress={() => {
//                                const link = document.createElement('a');
//                                link.href = photo.url;
//                                link.download = `photo_${index}.jpg`;
//                                document.body.appendChild(link);
//                                link.click();
//                                document.body.removeChild(link);
//                            }}>
//                                <Text style={{ color: 'blue', textAlign: 'center' }}>Download</Text>
//                            </TouchableOpacity>
//                        </View>
//                    ))}
//                </View>
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










export const DeviceView = React.memo((props: { device: BluetoothRemoteGATTServer }) => {
    const [subscribed, photos] = usePhotos(props.device);
    const agent = React.useMemo(() => new Agent(), []);
    const agentState = agent.use();

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
        // For draggable and resizable dialog
        const dialogPosition = React.useRef(new Animated.ValueXY({ x: 0, y: 0 })).current;
        const [dialogWidth, setDialogWidth] = React.useState(600);
        const [dialogHeight, setDialogHeight] = React.useState(600);
    
        // PanResponder for drag
        const panResponder = React.useMemo(() => PanResponder.create({
            onStartShouldSetPanResponder: () => true,
            onPanResponderMove: Animated.event([null, { dx: dialogPosition.x, dy: dialogPosition.y }], { useNativeDriver: false }),
            onPanResponderRelease: (_, gestureState) => {
                // Update position to final location
                dialogPosition.extractOffset();
                
            }
        }), []);
    
        // Font size based on dialog size
        const fontSize = Math.max(16, dialogWidth / 20);


        
    return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <View style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 }}>
                <TouchableOpacity style={{ backgroundColor: '#1f4888', padding: 10, borderRadius: 5, marginTop: 20 }} onPress={downloadAllPhotos}>
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
                                    <Text style={{ color: '#1f4888', textAlign: 'center' }}>Download</Text>
                                </TouchableOpacity>
                            </View>
                        ))}
                    </View>
                </ScrollView>
            </View>

            {/* Draggable and Resizable Dialog */}
            <Animated.View
                {...panResponder.panHandlers}
                style={{
                    backgroundColor: 'rgb(28 28 28)',
                    borderRadius: 64,
                    padding: 64,
                    position: 'absolute',
                    transform: [{ translateX: dialogPosition.x }, { translateY: dialogPosition.y }],
                    width: dialogWidth,
                    height: dialogHeight,
                }}
            >
                <View style={{ flexGrow: 1, justifyContent: 'center', alignItems: 'center' }}>
                    {agentState.loading && (<ActivityIndicator size="large" color={"white"} />)}
                    {agentState.answer && !agentState.loading && (
                        <ScrollView style={{ flexGrow: 1, flexBasis: 0 }}>
                            <Text style={{ color: 'white', fontSize: fontSize }}>{agentState.answer}</Text>
                        </ScrollView>
                    )}
                </View>

                <View style={{
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    marginLeft: 0,  // 设置与对话框左边的距离
                }}>
                    <TextInput
                        style={{
                            color: 'white',
                            height: dialogHeight * 0.1, // 根据对话框高度调整文本框高度
                            width: dialogWidth * 0.8,  // 根据对话框宽度调整文本框宽度
                            fontSize: Math.max(16, dialogHeight * 0.05), // 根据对话框高度动态调整字体大小
                            borderRadius: 16,
                            backgroundColor: 'rgb(48 48 48)',
                            padding: 6,
                            
                        }}
                        placeholder='How can I help you?'
                        placeholderTextColor={'#888'}
                        readOnly={agentState.loading}
                        onSubmitEditing={(e) => agent.answer(e.nativeEvent.text)}
                    />
                </View>

                {/* Resize handle */}
                <View
                    style={{
                        position: 'absolute',
                        right: 10,
                        bottom: 10,
                        width: 20,
                        height: 20,
                        backgroundColor: 'rgba(255, 255, 255, 0.2)',
                        borderRadius: 5,
                    }}
                    {...PanResponder.create({
                        onStartShouldSetPanResponder: () => true,
                        onPanResponderMove: (_, gestureState) => {
                            setDialogWidth(dialogWidth + gestureState.dx);
                            setDialogHeight(dialogHeight + gestureState.dy);
                        },
                    }).panHandlers}
                />
            </Animated.View>
        </View>
    );
});