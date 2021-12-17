import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
            'metaLoad': 'built_graph/tiny-yolo-3c.meta',
            'pbLoad': 'built_graph/tiny-yolo-3c.pb', 
            'labels': 'labels.txt',
            'threshold': 0.2,
            'gpu': 0.7
}

tfnet = TFNet(options)
frame = cv2.imread("0.jpg")
results = tfnet.return_predict(frame)

for color, result in zip(colors, results):
    tl = (result['topleft']['x'], result['topleft']['y'])
    br = (result['bottomright']['x'], result['bottomright']['y'])
    label = result['label']
    confidence = result['confidence']
    text = '{}: {:.0f}%'.format(label, confidence * 100)
    frame = cv2.rectangle(frame, tl, br, color, 5)
    frame = cv2.putText(
    frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    
cv2.imshow('frame', frame)
cv2.imwrite("frame.jpg", frame)


# options = {
#     'model': 'cfg/tiny-yolo-3c.cfg',
#     'load': 18875,
#     'threshold': 0.9,
#     'gpu': 0.7
# }

# tfnet = TFNet(options)
# print(tfnet)
# colors = [tuple(255 * np.random.rand(3)) for _ in range(300)]

# frame = cv2.imread("C:/Users/Sebastian Smit/Google Drive/champ foto's/testen nieuwe protoPC/22-1-13-51-2021/full_scan/stroke_4/scan.jpg")
# # C:\Users\Sebastian Smit\Google Drive\champ foto's\testen nieuwe protoPC\22-1-14-21-2021\full_scan\stroke_4
# frame2 = cv2.resize(frame , (860,640)) 
# cv2.imshow("frame", frame2)

# cv2.waitKey(0)

# results = tfnet.return_predict(frame)

# print(results)
# for color, result in zip(colors, results):
#     tl = (result['topleft']['x'], result['topleft']['y'])
#     br = (result['bottomright']['x'], result['bottomright']['y'])
#     label = result['label']
#     confidence = result['confidence']
#     text = '{}: {:.0f}%'.format(label, confidence * 100)
#     frame = cv2.rectangle(frame, tl, br, color, 5)
#     frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)


# frame2 = cv2.resize(frame , (1200,1040)) 
# cv2.imshow("frame", frame2)

# # cv2.imshow('frame', frame)
# cv2.waitKey(0)

# exit()


# import cv2
# from darkflow.net.build import TFNet
# import numpy as np
# import time

# class ClassName(object):
#     """docstring for ClassName"""
#     def __init__(self, arg):
#         super(ClassName, self).__init__()
#         self.arg = arg
#         self.options = 0
#         self.tfnet = 0
        

#     def Load_Tf(self):
#         options = {
#             'model': 'cfg/tiny-yolo-3c.cfg',
#             'load': 41525,
#             'threshold': 0.1,
#             'gpu': 1.0
#         }
#         self.tfnet = TFNet(options)


# colors = [tuple(255 * np.random.rand(3)) for _ in range(300)]

# frame = cv2.imread('000001.jpg')
# cv2.imshow("frame", frame)

# cv2.waitKey(0)

# results = tfnet.return_predict(frame)

# print(results)
# for color, result in zip(colors, results):
#     tl = (result['topleft']['x'], result['topleft']['y'])
#     br = (result['bottomright']['x'], result['bottomright']['y'])
#     label = result['label']
#     confidence = result['confidence']
#     text = '{}: {:.0f}%'.format(label, confidence * 100)
#     frame = cv2.rectangle(frame, tl, br, color, 5)
#     frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)


# # frame = cv2.resize(frame , (640,1000)) 
# cv2.imwrite("img.jpg", frame)
# cv2.imshow('frame', frame)
# cv2.waitKey(0)

