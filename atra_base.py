import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

def disp_img_fullscreen(img, name="test"):
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, 1)#Apparently we lost the cv2.cv.CV_WINDOW_FULLSCREEN part of opencv in some new release. Luckily, the equivalent is just setting this to 1
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_concatenated_row(samples):
    """
    Concatenate each sample in samples horizontally, along axis 1.
    Return the resulting array.
    """
    return np.concatenate([sample for sample in samples], axis=1)

def get_concatenated_col(samples):
    """
    Concatenate each sample in samples vertically, along axis 0.
    Return the resulting array.
    """
    return np.concatenate([sample for sample in samples], axis=0)

def graph_results(loss, acc):
    """
    Given loss and accuracy, plot both on two subplots
    """
    N = len(loss)
    x = np.linspace(0, N, N)
    plt.subplot(1,2,1)
    plt.plot(x, loss)
    plt.subplot(1,2,2)
    plt.plot(x,acc)
    plt.show()
