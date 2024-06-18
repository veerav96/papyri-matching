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

    # Find SIFT keypoints
    sift = cv2.SIFT_create()
    keypoints = sift.detect(warped, None)
    # Sort keypoints based on response strength
    keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)
    # Choose only the top 20 keypoints
    top_n_keypoints = keypoints_sorted[:20]
    # Draw keypoints on the warped image
    keypoints_img = cv2.drawKeypoints(warped, top_n_keypoints, None)
    cv2.namedWindow("Top 20 SIFT Keypoints", cv2.WINDOW_NORMAL)
    cv2.imshow("Top 20 SIFT Keypoints", keypoints_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Please click exactly 4 points.")

#TODO : KD TREES