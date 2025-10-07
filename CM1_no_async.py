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

def erosion(image,neighbors):
    print("starting erosion")
    (rows, cols) = image.shape
    eroded_image = numpy.empty_like(image)

    for i in range(rows):
        for j in range(cols):
            n = visit_neighbors(image, [i,j], neighbors, ERROSION)
            eroded_image[i][j] = min(n)
    print("ending erosion")
    return eroded_image

def dilation(image,neighbors):
    print("starting dilation")
    rows, cols = image.shape
    dilated_image = numpy.empty_like(image)

    for i in range(rows):
        for j in range(cols):
            n = visit_neighbors(image, [i,j], -neighbors, DILATION)
            dilated_image[i][j] = max(n)
    print("ending dilation")
    return dilated_image

def morphological_gradient(image):
    erroded = erosion(image,GAMMA4)
    dilated = dilation(image,GAMMA4)
    return dilated - erroded

def visit_neighbors(image, origin, neighbors, default):
    res = []
    for n in neighbors:
        i,j = [a + b for a, b in zip(origin, n)]
        if (i<0 or j<0 or i>=image.shape[0] or j>=image.shape[1]): res.append(default==DILATION)
        elif image[i][j] not in res: res.append(image[i][j])
    return res


if __name__ == "__main__":
    main()