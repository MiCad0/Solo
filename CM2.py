import asyncio
import time
import cv2
import numpy

ERROSION = 0
DILATION = 1
GAMMA4 = numpy.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
GAMMA8 = numpy.array([[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]])

def main():
    # Load an image
    image = cv2.imread('house.png')
    cv2.imshow('Image', image)
    cv2.waitKey(0)

    # Turn the image to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray_image)
    cv2.waitKey(0)

    print(f"started at {time.strftime('%X')}")
    gradient_image = morphological_gradient(gray_image)
    print(f"finished at {time.strftime('%X')}")
    cv2.imshow('Gradient Image', gradient_image)
    # gradient_image2 = morphological_gradient(gradient_image)          # Less sharp image
    # cv2.imshow('Gradient Image2', gradient_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_line(coords, index):
    rows = []
    for e in coords:
        elem = filter(lambda a: a[index] == e[index], coords)
        if elem not in rows: rows.append(elem)
    return e

def morphological_gradient(image):
    # To parrallelize we need to get a list of coords for each row
    # rows = get_line(GAMMA4,0)
    # cols = get_line(GAMMA4,1)
    eroded, dilated = numpy.zeros(image.shape),numpy.zeros(image.shape)
    print(eroded.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            eroded[i][j] = min(visit_neighbors(image, [i,j], GAMMA4, ERROSION))
            dilated[i][j] = max(visit_neighbors(image, [i,j], -GAMMA4, DILATION))
    return dilated-eroded

def visit_neighbors(image, origin, neighbors, default):
    res = set()
    for n in neighbors:
        i,j = [a + b for a, b in zip(origin, n)]
        if (i<0 or j<0 or i>=image.shape[0] or j>=image.shape[1]): res.add(default==DILATION and 255 or 0 )
        else: res.add(image[i][j])
    return res

if __name__ == "__main__":
    main()