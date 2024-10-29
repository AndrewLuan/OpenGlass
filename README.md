# **OpenGlass**

###  准备工作

**项目地址：**https://github.com/BasedHardware/OpenGlass

硬件：

- [Seeed Studio XIAO ESP32 S3 Sense](https://www.amazon.com/dp/B0C69FFVHH/ref=dp_iou_view_item?ie=UTF8&psc=1)（带蓝牙、摄像头）

- [EEMB LP502030 3.7v 250mAH battery](https://www.amazon.com/EEMB-Battery-Rechargeable-Lithium-Connector/dp/B08VRZTHDL)

- [3D printed glasses mount case](https://storage.googleapis.com/scott-misc/openglass_case.stl) recommended：光敏树脂打印，否则合不上盖子

软件：

#### **1 .下载Arduino IDE**

https://www.arduino.cc/en/software

#### **2.设置ESP32S3主板程序**

Open Terminal

`git clone https://github.com/BasedHardware/OpenGlass.git`

(也可以手动下载)

然后打开找到OpenGlass文件夹里的 [firmware folder](https://github.com/BasedHardware/openglass/tree/main/firmware) 文件夹，并在 Arduino IDE 中打开 `firmware.ino` 文件。

**为ESP32S3 Sense主板设置 Arduino IDE：**

顶部菜单栏>>Arduino IDE>>Settings, 然后在**“其他 Boards Manager URL”("Additional Boards Manager URLs" )**填写这个链接： `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`

![](/Users/andrewluan/Documents/Screenshot 2024-07-19 at 17.24.00.png)

导航到**“工具”>“开发板”>“开发板管理器...”（Tools > Board > Boards Manager）**，在搜索框中输入关键字 `esp32` ，下第二个的2.0.17版本。最近网络不稳定导致下载时间极长，建议手动下载。（github上有包）

![](/Users/andrewluan/Documents/Screenshot 2024-07-19 at 17.32.14.png)

esp32安装完成后，就可以在 Arduino IDE 顶部选择开发板和端口。弹窗中搜索 `xiao` 并选择 `XIAO_ESP32S3`。

点击左边菜单的库，搜索并安装 `ArduinoBLE`

![](/Users/andrewluan/Documents/Screenshot 2024-07-19 at 17.36.04.png)

在刷机之前，转到Arduino IDE中的“工具”下拉列表，并确保将“PSRAM：”设置为“PSRAM：”“OPI PSRAM”

![](/Users/andrewluan/Documents/Screenshot 2024-07-19 at 17.34.31.png)

以上步骤做完就ok了

**现在就可以，用一个type c数据线把板子和电脑连接，就可以进行烧录程序了。**

点击 `Verify` 按钮（对号图标），检查代码是否有错误。

点击 `Upload` 按钮（箭头图标），将代码上传到XIAO ESP32S3开发板。

==注意信号线要是usb-typeC的，不能两头typeC。对于只有typeC口的电脑需用扩展坞。==

对于Mac电脑要装驱动程序

![](/Users/andrewluan/Documents/Screenshot 2024-07-19 at 17.42.52.png)

About this Mac>>More Info>>System Report>>USB, 检查串口是否正常

![](/Users/andrewluan/Documents/Screenshot 2024-07-19 at 17.45.59.png)

不能选Bluetooth那个。

1. 1. #### 安装Expo

   2. Expo，这是一个用于构建React Native应用的工具。需要先安装Expo CLI，然后才能启动你的项目

   3. 以下是安装和使用Expo的步骤：

   4. 1. 打开终端。

      2. 运行以下命令来全局安装Expo CLI：`npm install -g expo-cli`

      3. 验证安装： `expo --version`

         

#### **04 完善硬件：XIAO ESP32S3开发板连接天线、摄像头、电池** 

**天线的安装**

在XIAO ESP32S3的正面左下角，巧妙地设计了一个独立的“WiFi/BT天线连接器”。为了捕获更清晰的WiFi和蓝牙信号，需要从包装中取出附带的天线，并细心地将其安装至连接器上。

**安装扩展板（用于Sense）**

XIAO ESP32S3 Sense，还包括一个扩展板。此扩展板具有1600*1200 OV2640摄像头传感器、板载SD卡插槽和数字麦克风。

通过使用XIAO ESP32S3 Sense安装扩展板，您可以使用扩展板上的功能。

安装扩展板非常简单，只需将扩展板上的连接器与XIAO ESP32S3上的B2B连接器对齐，用力按压并听到“咔嗒”一声，即可完成安装。

**焊接电池**

如图所示。需要额外安装开关，否则电路板会过热。

![](/Users/andrewluan/Downloads/WechatIMG38.jpg)

#### **05 启动照片的页面程序**

烧录完主板，还需要设置一下 Groq 和OpenAI的API。

在位于 https://github.com/BasedHardware/OpenGlass/blob/main/sources/keys.ts `keys.ts` 的文件中添加 Groq 和 OpenAI 的 API 密钥。其中groq可进官网免费申领（需要挂梯子否则会被ban），OpenAI为optional。

对于 Ollama，从存储库中自行托管 https://github.com/ollama/ollama 的 REST API，并将 URL 添加到 `keys.ts` 文件中。

找到openglass文件夹并右键new terminal at folder

**启动应用程序：**`npm start`

启动成功，你在terminal能看到web页面地址，复制到浏览器可以打开。

![](/Users/andrewluan/Documents/Screenshot 2024-07-19 at 21.54.26.png)

1. 1. 

### other settings 
1. 

   Hey @here I saw many people had issues with the repo cuz last model was depreciated You should have just changed moondream version and it would work  I just updated github and recorded a video here is the video of it working: https://www.loom.com/share/4c5666ef283f4b33b4705a21c71fc461 How to make it work: 

   1. do git pull from github
   2. install ollama and then go to terminal and type "ollama pull moondream:1.8b-v2-fp16"
   3. paste "http://localhost:11434/api/chat" in keys.ts as ollama key
   4. Provide  + openai key + groq

   Thanks to @SG Ramanan for the help(edited)

   [Loom](https://www.loom.com/)

   [Arc - 13 June 2024](https://www.loom.com/share/4c5666ef283f4b33b4705a21c71fc461)

   

   ![Image](https://images-ext-1.discordapp.net/external/ZNK5PJDLJzg_TsUdWHlSDLt5hQOboOkgNcuUjNOPjpk/https/cdn.loom.com/sessions/thumbnails/4c5666ef283f4b33b4705a21c71fc461-00001.gif?width=160&height=104)

2.**Chatgpt Api Key**

https://chatanywhere.apifox.cn/

https://github.com/chatanywhere/GPT_API_free?tab=readme-ov-file

但每天有次数限制

授权成功
您的免费API Key为: sk-DXVsSf9w131mIhg4LLG6tY0n0dlsotDPTlWcPRiZJS59RwXq
请妥善保管，不要泄露给他人，如泄漏造成滥用可能会导致Key被封禁
## TroubleShooting

板子在serial monitor重复报错可boot后上传示例文档如blink，再上传一遍firmware

