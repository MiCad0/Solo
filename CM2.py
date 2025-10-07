from CM1_no_async import *

def main():
    # Load an image
    image = cv2.imread('rice.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # cv2.imshow('Image', gray_image)
    # cv2.waitKey(0)
    # print(B(2))
    print(f"started at {time.strftime('%X')}")
    botHat = bottom_hat(gray_image,numpy.array(list(B(10))))
    print(f"finished at {time.strftime('%X')}")
    cv2.imwrite('output.png', botHat)
    # botHat = cv2.imread('rice.png')
    # botHat = cv2.cvtColor(botHat, cv2.COLOR_BGR2GRAY) 
    cv2.imshow('rice', threshold(botHat,240))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def opening(image, neighbors):
    print("Starting opening")
    return dilation(erosion(image,neighbors),neighbors)

def closing(image, neighbors):
    print("Starting closing")
    return erosion(dilation(image, neighbors),neighbors)

def top_hat(image,neighbors):
    return image - opening(image,neighbors)

def bottom_hat(image,neighbors):
    return closing(image,neighbors) - image

def B(n):
    x, y = numpy.mgrid[-n+1:n, -n+1:n]
    return numpy.stack((x.ravel(), y.ravel()), axis=-1)

def threshold(image,thresh):
    print(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i][j] > thresh): 
                image[i][j] = 255 
            else:
                image[i][j] = 0
    return image

if __name__ == "__main__":
    main()