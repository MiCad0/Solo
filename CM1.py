import cv2
import numpy

ERROSION = 0
DILATATION = 1

def main():
    # Load an image
    image = cv2.imread('house.png')
    cv2.imshow('Image', image)
    cv2.waitKey(0)

    # Turn the image to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray_image)
    cv2.waitKey(0)


    gradient_image = morphological_gradient(gray_image)
    cv2.imshow('Gradient Image', gradient_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def errosion(image):
    rows, cols = image.shape
    erroded_image = image.copy()
    gamma4 = numpy.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    gamma8 = numpy.array([[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]])

    for i in range(rows):
        for j in range(cols):
            n = visit_neighbors(image, [i,j], gamma4, ERROSION)
            erroded_image[i][j] = min(n)
    return numpy.asarray(erroded_image)

def dilation(image):
    rows, cols = image.shape
    erroded_image = image.copy()
    gamma4 = numpy.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    gamma8 = numpy.array([[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]])

    for i in range(rows):
        for j in range(cols):
            n = visit_neighbors(image, [i,j], -gamma4, DILATATION)
            erroded_image[i][j] = max(n)
    return numpy.asarray(erroded_image)

def morphological_gradient(image):
    erroded = errosion(image)
    dilated = dilation(image)
    return dilated - erroded

def visit_neighbors(image, origin, neighbors, default):
    res = set()
    for n in neighbors:
        i,j = [a + b for a, b in zip(origin, n)]
        if (i<0 or j<0 or i>=image.shape[0] or j>=image.shape[1]): res.add(default==DILATATION and 255 or 0 )
        else: res.add(image[i][j])
    return res


if __name__ == "__main__":
    main()