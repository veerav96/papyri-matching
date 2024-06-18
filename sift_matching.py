import cv2
import os
import pickle
import time
import numpy as np

# Global variables to store points
points = []
#SO THAT USER CAN CROP THE IMAGE
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


def load_images_from_folder(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            full_path = os.path.join(root, filename)
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Construct the key
                key = os.path.join(os.path.relpath(root, folder), os.path.splitext(filename)[0])
                images.append((key, img))
    return images


def compute_sift_features(image,max_keypoints=1000):
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image)
    # Sort keypoints based on their strength
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_keypoints]
    keypoints, descriptors = sift.compute(image, keypoints)
    return keypoints, descriptors

def update_reference_features(reference_images_folder, features_file):
    # Load existing features if they exist
    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            features = pickle.load(f)
    else:
        features = {}
    
    # Load reference images
    reference_images = load_images_from_folder(reference_images_folder)
    
    # Compute features for new images
    new_features = {}
    for filename, img in reference_images: #subfolder/filename
        if filename not in features:
            keypoints, descriptors = compute_sift_features(img)        
            new_features[filename] = descriptors

    # Update the features dictionary
    features.update(new_features)
    
    # Save updated features
    with open(features_file, 'wb') as f:
        pickle.dump(features, f)

    return new_features.keys()  # Return the names of new images added

def load_precomputed_features(features_file):
    with open(features_file, 'rb') as f:
        return pickle.load(f)

def sift_feature_matching(query_kp, query_des, reference_features):
    start_time = time.time()
    index_params = dict(algorithm=1, trees=10)  # FLANN parameters
    search_params = dict(checks=50)  # Higher checks for better precision
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    best_match = None
    max_good_matches = 0
    best_kp2 = None
    best_matches = None

    for filename, des2 in reference_features.items():
        if des2 is not None:
            matches = flann.knnMatch(query_des, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match = filename
                best_kp2 = des2
                best_matches = good_matches
    
    time_taken = time.time() - start_time
    print("Time taken for matching:", time_taken, "seconds")
    
    return best_match, best_matches

def main():
    query_image_path = '/Users/cprao/Desktop/heidelberg_SEM3/practical/papyri-matching/junk/p_g_107_0001.jpg'
    
    
    reference_images_folder = '/Users/cprao/Desktop/heidelberg_SEM3/practical/papyri-matching/images/'
    features_file = 'reference_features.pkl'

    print("Updating reference features...")
    new_images = update_reference_features(reference_images_folder, features_file)

    if new_images:
        print(f"Added new reference images: {new_images}")
    else:
        print("No new images were added.")

    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = query_image.shape[:2]

    cv2.namedWindow("User Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("User Image", width, height)
    cv2.imshow("User Image", img)
    cv2.setMouseCallback("User Image", click_event)
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

    
        query_kp, query_des = compute_sift_features(query_image)
        
        reference_features = load_precomputed_features(features_file)
    
        best_match, best_matches = sift_feature_matching(query_kp, query_des, reference_features)
        
        if best_match:
            print(f"The best matching image is: {best_match}")
            #VISUALIZATION
            # Load the best matching reference image
            best_match_image_path = os.path.join(reference_images_folder, best_match)+".jpg"
            best_match_image = cv2.imread(best_match_image_path, cv2.IMREAD_GRAYSCALE)
            best_kp2, _ = compute_sift_features(best_match_image)

            # Draw matches
            top_matches = best_matches[:50]
            match_img = cv2.drawMatchesKnn(query_image, query_kp, best_match_image, best_kp2, [top_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Display the matched keypoints
            height, width = match_img.shape[:2]
            cv2.namedWindow("Matched keypoints", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Matched keypoints", width, height)
        
            cv2.imshow('Matched keypoints', match_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No match found.")
    else :
        print("Error: Please click exactly 4 points.")

if __name__ == "__main__":
    main()
