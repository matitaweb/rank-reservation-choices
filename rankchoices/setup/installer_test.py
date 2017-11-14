import cv2

try:
    import cv2
except ImportError:
    print "ERROR install"
    pass

print "OPENCV insted"
print cv2.__version__