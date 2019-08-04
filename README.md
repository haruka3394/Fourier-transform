# ディジタル信号処理と画像処理　フーリエ変換

B173394 

- OpenCVを用いてフーリエ変換・逆フーリエ変換をリアルタイムに行う. マウスで周波数を指定する. 

以下にソースコードを示す. まずAnacondaをインストールし, JupyterにてOpenCVをインストールした. 

``` Python
import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('flower.jpg',0)
img = cv2.resize(im,(240,180))

clicklog = np.zeros(im.shape)

# フーリエ変換
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# 逆フーリエ変換
dft_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(dft_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])


def addClick(x,y,click):
A = click
A[y,x] = 1
return A

def mkWave(x, y):
h = np.zeros(im.shape)
h[y, x] = 1
wave = cv2.idft(h)
return wave

def reset(x, y, A):
y = np.zeros(dft_shift.shape)
y[:,:,0] = A 
y[:,:,1] = A 
copy = dft_shift*y
copy = np.fft.ifftshift(copy)
copy = cv2.idft(copy)
re = cv2.magnitude(copy[:,:,0],copy[:,:,1])
return re

def click(event):
global clicklog, X, Y

if event.button == 1:
X = int(round(event.xdata))
Y = int(round(event.ydata))

ax7.plot(X, Y, marker='.', markersize='1')

clicklog = addClick(X, Y, clicklog)

def key(event):
if event.key == 'q':
sys.exit()

def release(event):
global clicklog, X, Y
wave = mkWave(X, Y)
ax6.imshow(wave, cmap='gray')

reim = reset(X, Y, clicklog)
reim = np.float32(reim)
ax4.imshow(reim, cmap='gray')

plt.draw()

fig = plt.figure(figsize=(9,4))

ax1 = fig.add_subplot(2,3,1)
ax1.imshow(im, cmap='gray')
ax1.set_title('Input Image')
ax1.set_xticks([]), ax1.set_yticks([])

ax2 = fig.add_subplot(2,3,2)
ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.set_title('Magnitude Spectrum')
ax2.set_xticks([]), ax2.set_yticks([])

ax3 = fig.add_subplot(2,3,3)
ax3.imshow(img_back, cmap='gray')
ax3.set_title('IFFT')
ax3.set_xticks([]), ax3.set_yticks([])

ax4 = fig.add_subplot(2,3,4)

ax6 = fig.add_subplot(2,3,6)

wind2=plt.figure(figsize=(8,4))

ax7 = wind2.add_subplot(1,1,1)
ax7.imshow(magnitude_spectrum, cmap='gray')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0)

wind2.canvas.mpl_connect('button_press_event', click)
wind2.canvas.mpl_connect('motion_notify_event', click)
wind2.canvas.mpl_connect('button_release_event', release)
wind2.canvas.mpl_connect('key_press_event', key)

```
- コードの説明
- cv2 (openCVを使用する), numpy as np (array型の変数を使用する),  matplotlib.pyplot as plt (グラフを表示させる)ライブラリをインポートする. 

- `y = []` : 配列を得ている
- `capture = cv2.VideoCapture(0)` : カメラのキャプチャを開始させる. 
``` Phython
while(True):
ret, frame = capture.read()
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
avr = np.average(gray)
print(avr)
y.append(avr)
cv2.imshow('frame',gray)
if cv2.waitKey(1) & 0xFF == ord('q'):
break
```
- このwhile文でキーボードからqを入力されるまで画像を取得して基礎値の平均を取っている. これは毎フレームごとに行われる. 
- `capture.read()` : カメラから画像を取得する. それをframeに入れている. 
- `gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)` : 画像をグレースケールに変換している. 
- `avr = np.average(gray)` : グレースケールに変換した画像の輝度値の平均値を取っている.  
- `print(avr)` : 輝度値の平均値を表示させている. 
- `y.append(avr)` : yに輝度値の平均値の値を入れて行っている. 
- `cv2.imshow('frame',gray)` : カメラの画像をウィンドウに出力している. 

- `print(len(y))` : 輝度値の平均値がいくつあるかを表示させている. 
- `x=np.linspace(1,100,len(y))` : xを定義している. 1から100までを輝度値の平均値の個数で割っている. 
- `capture.release(), cv2.destroyAllWindows()` : カメラから得ているキャプチャを終了し, ウィンドウを閉じる. 
- `plt.plot(x, y, label="test"), plt.show()` : w,yの値を表示させている. 


- 実行結果  
今回の課題は, カメラに指を押し当て脈拍を図るというものだった. これは指に光をあてることで指を透過し, 光の微量な変化を読み込み, 輝度値の平均としてグラフに表示している. 以下にグラフを表示する. 

![](opencv-2.pdf)

これにより脈拍をグラフの一定の変化により確認することができる. また, 輝度値の値をkidoti.txtに示し, ウィンドウの画像の変化をGIFに示す. GIFファイルからも光の微量な変化を見ることができる. 

![](openCV.gif)

- バージョン
- macOS Mojave 10.14.5
- Pyhton 3.7

- 参考文献
- [フーリエ変換OpenCV](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html)  
フーリエ変換をする方法

- [matplotlibによるデータ可視化の方法](https://qiita.com/ynakayama/items/8d3b1f7356da5bcbe9bc）  
データを可視化する

- [PythonとOpenCVで画像処理](http://rasp.hateblo.jp/entry/2016/01/24/204539)  
- [Python+OpenCVでMouseイベントを取得してお絵描きをする話](https://ensekitt.hatenablog.com/entry/2018/06/17/200000))
マウスイベントについて

