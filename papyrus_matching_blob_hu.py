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
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),borderMode=cv2.BORDER_TRANSPARENT)

    return warped

# Script
#img = cv2.imread('/Users/cprao/Desktop/heidelberg_SEM3/practical/junk/G0107R.jpg')
img = cv2.imread('/Users/cprao/Desktop/heidelberg_SEM3/practical/papyri-matching/query_images/19317_p_g_192_c.jpg')
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
    blurred_image = cv2.GaussianBlur(gray_warped, (3, 3), 0)
    blurred_image = cv2.medianBlur(blurred_image, 3)
    _, binary_img = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Display the binary image in a resizable window
    cv2.namedWindow('Binary Image (Otsu)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Binary Image (Otsu)', warped_width, warped_height)
    cv2.imshow('Binary Image (Otsu)', binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the outermost contour (the outer boundary)
    outer_contour = max(contours, key=cv2.contourArea)

    # Filter out the outer boundary contour and keep only the interior holes
    interior_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < cv2.contourArea(outer_contour)]

    # Sort the interior contours based on area
    interior_contours = sorted(interior_contours, key=cv2.contourArea, reverse=True)[:10]

    # Draw interior contours on the image
    contour_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, interior_contours, -1, (0, 255, 0), 2)

    # Display the image with contours
    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Contours', warped_width, warped_height)
    cv2.imshow('Contours', contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Compute Hu moments for each interior contour
    for i, contour in enumerate(interior_contours):
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        print(f"Contour {i + 1}: Hu Moments = {hu_moments}")

else:
    print("Error: Please click exactly 4 points.")

# Wait for the user to press Enter before exiting
#input("Press Enter to exit...")
