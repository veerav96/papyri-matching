import cv2
import os
import pickle
import time
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
        cv2.imshow("User Image", img)

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
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), borderMode=cv2.BORDER_TRANSPARENT)

    return warped

def load_images_from_folder(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            full_path = os.path.join(root, filename)
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                key = os.path.join(os.path.relpath(root, folder), os.path.splitext(filename)[0])
                images.append((key, img))
    return images

def compute_sift_features(image, max_keypoints=1000):
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image)
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_keypoints]
    keypoints, descriptors = sift.compute(image, keypoints)
    return keypoints, descriptors


def keypoints_to_list(keypoints):
    keypoints_list = []
    for kp in keypoints:
        kp_info = {
            'pt': kp.pt,
            'size': kp.size,
            'angle': kp.angle,
            'response': kp.response,
            'octave': kp.octave,
            'class_id': kp.class_id
        }
        keypoints_list.append(kp_info)
    return keypoints_list



def list_to_keypoints(keypoints_list):
    keypoints = []
    for kp_info in keypoints_list:
        x, y = kp_info['pt']  # Extract x, y coordinates
        size = kp_info['size']
        angle = kp_info['angle']
        response = kp_info['response']
        octave = kp_info['octave']
        class_id = kp_info['class_id']
        
        # Create KeyPoint object
        keypoint = cv2.KeyPoint(x=x, y=y, size=size, angle=angle,
                                response=response, octave=octave,
                                class_id=class_id)
        keypoints.append(keypoint)
    return keypoints


def update_reference_features(reference_images_folder, features_file):
    if os.path.exists(features_file):
        with open(features_file, 'rb') as f:
            features = pickle.load(f)
    else:
        features = {}

    reference_images = load_images_from_folder(reference_images_folder)

    new_features = {}
    for filename, img in reference_images:
        if filename not in features:
            keypoints, descriptors = compute_sift_features(img)
            keypoints_list = keypoints_to_list(keypoints)
            new_features[filename] = (keypoints_list, descriptors)

    features.update(new_features)

    with open(features_file, 'wb') as f:
        pickle.dump(features, f)

    return new_features.keys()


def load_precomputed_features(features_file):
    with open(features_file, 'rb') as f:
        features = pickle.load(f)
    
    for filename in features:
        keypoints_list, descriptors = features[filename]
        keypoints = list_to_keypoints(keypoints_list)
        features[filename] = (keypoints, descriptors)
    
    return features


def sift_feature_matching(query_kp, query_des, reference_features, top_n=5):
    start_time = time.time()
    index_params = dict(algorithm=1, trees=20)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    best_matches_info = []

    for filename, (ref_kp, des2) in reference_features.items():
        if des2 is not None:
            matches = flann.knnMatch(query_des, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good_matches) >= 4:  # Minimum 4 points required to apply RANSAC
                src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Apply RANSAC to find the homography matrix and filter matches
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if mask is not None:
                    matchesMask = mask.ravel().tolist()
                    inliers = [good_matches[i] for i in range(len(matchesMask)) if matchesMask[i] == 1]
                    best_matches_info.append((filename, len(inliers), inliers))

    best_matches_info.sort(key=lambda x: x[1], reverse=True)
    top_matches = best_matches_info[:top_n]
    time_taken = time.time() - start_time
    print("Time taken for matching:", time_taken, "seconds")
    return top_matches

    
#RANSAC incorporating orientation as well
'''
def sift_feature_matching(query_kp, query_des, reference_features, top_n=5):
    start_time = time.time()
    index_params = dict(algorithm=1, trees=20)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    best_matches_info = []

    for filename, (ref_kp, des2) in reference_features.items():
        if des2 is not None:
            matches = flann.knnMatch(query_des, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good_matches) >= 4:  # Minimum 4 points required to apply RANSAC
                # Extract keypoints and descriptors
                src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                src_angles = np.float32([query_kp[m.queryIdx].angle for m in good_matches])
                dst_angles = np.float32([ref_kp[m.trainIdx].angle for m in good_matches])

                # Combine positions and orientations
                src_pts_oriented = np.hstack((src_pts, src_angles[:, np.newaxis]))
                dst_pts_oriented = np.hstack((dst_pts, dst_angles[:, np.newaxis]))

                # Apply RANSAC to find the homography matrix and filter matches
                M, mask = cv2.findHomography(src_pts_oriented, dst_pts_oriented, cv2.RANSAC, 5.0)

                if mask is not None:
                    matchesMask = mask.ravel().tolist()
                    inliers = [good_matches[i] for i in range(len(matchesMask)) if matchesMask[i] == 1]
                    best_matches_info.append((filename, len(inliers), inliers))

    best_matches_info.sort(key=lambda x: x[1], reverse=True)
    top_matches = best_matches_info[:top_n]
    time_taken = time.time() - start_time
    print("Time taken for matching:", time_taken, "seconds")
    return top_matches
'''


def process_query_image(query_image_path):
    global img
    img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]

    cv2.namedWindow("User Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("User Image", width, height)
    cv2.imshow("User Image", img)
    cv2.setMouseCallback("User Image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 4:
        pts = np.array(points, dtype="float32")
        warped = perspective_transform(img, pts)
        height, width = warped.shape[:2]
        cv2.namedWindow("warped", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("warped", width, height)
        cv2.imshow("warped", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cropped_warped_image = crop_image(warped,5) # remove border
        return cropped_warped_image
        
    else:
        print("Error: Four points were not selected.")
        return None

#remove white boundary which opencv creates after warping
def crop_image(image, pixels):
    h, w = image.shape[:2]
    cropped_image = image[pixels:h-pixels, pixels:w-pixels]
    return cropped_image

def display_and_match(query_image, reference_images_folder, features_file, top_n=5):
    if query_image is not None:
        query_kp, query_des = compute_sift_features(query_image)

        reference_features = load_precomputed_features(features_file)

        top_matches = sift_feature_matching(query_kp, query_des, reference_features, top_n=top_n)

        for idx, (filename, num_matches, good_matches) in enumerate(top_matches, start=1):
            print(f"Top {idx} Match - Filename: {filename}, Num Matches: {num_matches}")

            best_match_image_path = os.path.join(reference_images_folder, filename) + ".jpg"
            best_match_image = cv2.imread(best_match_image_path, cv2.IMREAD_GRAYSCALE)
            best_kp2, _ = compute_sift_features(best_match_image)  # get keypoints only

            match_img = cv2.drawMatches(query_image, query_kp, best_match_image, best_kp2, good_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            height, width = match_img.shape[:2]
            cv2.namedWindow(f"Matched keypoints - Top {idx}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Matched keypoints - Top {idx}", width, height)
            cv2.imshow(f"Matched keypoints - Top {idx}", match_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        print("Error: Query image is None.")





def main():
    query_image_path = '/Users/cprao/Desktop/heidelberg_SEM3/practical/papyri-matching/query_images/19317_p_g_192_c.jpg'
    reference_images_folder = '/Users/cprao/Desktop/heidelberg_SEM3/practical/papyri-matching/images/'
    features_file = 'sift_features_with_keypoints.pkl'

    print("Updating reference features...")
    new_images = update_reference_features(reference_images_folder, features_file)

    if new_images:
        print(f"Added new reference images: {new_images}")
    else:
        print("No new images were added.")

    query_image = process_query_image(query_image_path)
    display_and_match(query_image, reference_images_folder, features_file)

if __name__ == "__main__":
    main()
    exit
