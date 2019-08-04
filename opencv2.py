import numpy as np
import cv2
from matplotlib import pyplot as plt

# 画像の読み込み
img = cv2.imread('flower.jpg',0)

img = cv2.resize(im,(240,180))

# clicklogの初期化
clicklog = np.zeros(im.shape)

# フーリエ変換
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# 逆フーリエ変換
dft_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(dft_ishift)
img_back = cv2.magnitude(img_back[:,:,0],im  g_back[:,:,1])

def addClick(x,y,click):
    A = click
    A[y,x] = 1
    return A

# sin波の切り抜き
def mkWave(x, y):
    h = np.zeros(im.shape)
    h[y, x] = 1
    wave = cv2.idft(h)
    return wave

# sin波の合成
def reset(x, y, A):
    y = np.zeros(dft_shift.shape)
    y[:,:,0] = A
    y[:,:,1] = A
    copy = dft_shift*y
    copy = np.fft.ifftshift(copy)
    copy = cv2.idft(copy)
    re = cv2.magnitude(copy[:,:,0],copy[:,:,1])
    return re

# クリック時の処理
def click(event):
    global clicklog, X, Y
    if event.button == 1:
        X = int(round(event.xdata))
        Y = int(round(event.ydata))
        
        ax7.plot(X, Y, marker='.', markersize='1')
        
        clicklog = addClick(X, Y, clicklog)

# qを押したら終了
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

# マウスイベント
wind2.canvas.mpl_connect('button_press_event', click)
wind2.canvas.mpl_connect('motion_notify_event', click)
wind2.canvas.mpl_connect('button_release_event', release)
wind2.canvas.mpl_connect('key_press_event', key)
