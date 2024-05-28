import cv2
import numpy as np

# Global variables to store points
points = []

def click_event(event, x, y, flags, params):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(points) == 4:
            ordered_pts = order_points(np.array(points, dtype="float32"))
            cv2.polylines(img, [np.int32(ordered_pts)], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.imshow("Image", img)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# Script
img = cv2.imread('/Users/cprao/Desktop/heidelberg_SEM3/practical/junk/G0107R.jpg')
height, width = img.shape[:2]

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", width, height)
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) == 4:
    pts = np.array(points, dtype="float32")
    warped = perspective_transform(img, pts)

    warped_height, warped_width = warped.shape[:2]

    cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Warped Image", warped_width, warped_height)
    cv2.imshow("Warped Image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('warped_image.jpg', warped)

    # Convert warped image to grayscale
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    _, binary_img = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Display the binary image in a resizable window
    cv2.namedWindow('Binary Image (Otsu)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Binary Image (Otsu)', warped_width, warped_height)
    cv2.imshow('Binary Image (Otsu)', binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Blob detection
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100

    # Filter by Circularity
    params.filterByCircularity = False

    # Filter by Convexity
    params.filterByConvexity = False

    # Filter by Inertia
    params.filterByInertia = False

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(binary_img)

    # Draw blobs on the binary image
    blobs_img = cv2.drawKeypoints(binary_img, keypoints, np.array([]), (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the binary image with detected blobs
    cv2.namedWindow('Binary Image with Blobs', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Binary Image with Blobs', warped_width, warped_height)
    cv2.imshow('Binary Image with Blobs', blobs_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Compute Hu moments for each blob
    for i, keypoint in enumerate(keypoints):
        moments = cv2.moments(keypoint.pt)
        hu_moments = cv2.HuMoments(moments).flatten()
        print("Blob {}: Hu Moments = {}".format(i + 1, hu_moments))

else:
    print("Error: Please click exactly 4 points.")



# Wait for the user to press Enter before exiting
input("Press Enter to exit...")
