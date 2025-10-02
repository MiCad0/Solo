import asyncio
import time
import cv2
import numpy
import threading

ERROSION = 0
DILATION = 1

async def main():
    # Load an image
    image = cv2.imread('house.png')
    cv2.imshow('Image', image)
    cv2.waitKey(0)

    # Turn the image to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray_image)
    cv2.waitKey(0)

    print(f"started at {time.strftime('%X')}")
    gradient_image = await morphological_gradient(gray_image)
    print(f"finished at {time.strftime('%X')}")
    cv2.imshow('Gradient Image', gradient_image)
    # gradient_image2 = morphological_gradient(gradient_image)          # Less sharp image
    # cv2.imshow('Gradient Image2', gradient_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

async def errosion(image):
    rows, cols = image.shape
    erroded_image = image.copy()
    gamma4 = numpy.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    gamma8 = numpy.array([[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]])

    for i in range(rows):
        for j in range(cols):
            n = visit_neighbors(image, [i,j], gamma4, ERROSION)
            erroded_image[i][j] = min(n)
    return numpy.asarray(erroded_image)

async def dilation(image):
    rows, cols = image.shape
    erroded_image = image.copy()
    gamma4 = numpy.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    gamma8 = numpy.array([[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]])

    for i in range(rows):
        for j in range(cols):
            n = visit_neighbors(image, [i,j], -gamma4, DILATION)
            erroded_image[i][j] = max(n)
    return numpy.asarray(erroded_image)

async def morphological_gradient(image):
    erroded = asyncio.create_task(errosion(image))
    dilated = asyncio.create_task(dilation(image))
    return await dilated - await erroded

def visit_neighbors(image, origin, neighbors, default):
    res = set()
    for n in neighbors:
        i,j = [a + b for a, b in zip(origin, n)]
        if (i<0 or j<0 or i>=image.shape[0] or j>=image.shape[1]): res.add(default==DILATION and 255 or 0 )
        else: res.add(image[i][j])
    return res


if __name__ == "__main__":
    asyncio.run(main())