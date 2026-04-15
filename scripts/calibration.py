import cv2
import numpy as np
import glob

# ==============================
# SETTINGS (EDIT THESE)
# ==============================

CHECKERBOARD = (10, 7)  # inner corners (cols, rows)
square_size = 0.025  # meters (e.g. 25mm = 0.025m)

image_folder = (
    "/Users/dan/Code/kth/dd2419_ws/bags/calibration/*.png"  # your calibration images
)
# test_image_path = "test.jpg"  # image to undistort

# test_image_path = "/Users/dan/Code/kth/dd2419_ws/bags/box_images/2.png"
test_image_path = "/Users/dan/Code/kth/dd2419_ws/bags/calibration/image3.png"

# ==============================
# PREPARE OBJECT POINTS
# ==============================

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image

# ==============================
# LOAD IMAGES
# ==============================

images = glob.glob(image_folder)

if len(images) == 0:
    print("No images found!")
    exit()

# ==============================
# FIND CHECKERBOARD CORNERS
# ==============================

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if ret:
        # refine corners
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (3, 3),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        )

        objpoints.append(objp)
        imgpoints.append(corners)

        print(f"Detected corners in {fname}")
    else:
        print(f"FAILED to detect corners in {fname}")

# ==============================
# CALIBRATION
# ==============================

N_OK = len(objpoints)
print(f"\nValid images: {N_OK}")

if N_OK < 5:
    print("Not enough valid images!")
    exit()

img_shape = gray.shape[::-1]

K = np.zeros((3, 3))
D = np.zeros((4, 1))

rvecs = []
tvecs = []

rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    img_shape,
    K,
    D,
    rvecs,
    tvecs,
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
)

print("\nCalibration done!")
print("RMS error:", rms)
print("K (camera matrix):\n", K)
print("D (distortion):\n", D)

# Save parameters
np.savez("calibration_fisheye.npz", K=K, D=D)

# ==============================
# UNDISTORT TEST IMAGE
# ==============================

# img = cv2.imread(test_image_path)
# h, w = img.shape[:2]

# map1, map2 = cv2.fisheye.initUndistortRectifyMap(
#     K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
# )

# undistorted = cv2.remap(
#     img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
# )

# cv2.imshow("Original", img)
# cv2.imshow("Undistorted", undistorted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.imread(test_image_path)
h, w = img.shape[:2]
scale = 1.0  # increase to preserve more content (try 1.2 → 2.0)
balance = 1.0  # 1.0 = keep all pixels

new_size = (int(w * scale), int(h * scale))

# ==============================
# COMPUTE NEW CAMERA MATRIX
# ==============================

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3), balance=balance, new_size=new_size
)

# ==============================
# CREATE REMAP
# ==============================

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, new_size, cv2.CV_16SC2
)

# ==============================
# UNDISTORT
# ==============================

undistorted = cv2.remap(
    img,
    map1,
    map2,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_REFLECT_101,  # <-- fills missing with mirrored pixels
)

# ==============================
# SHOW RESULT
# ==============================

cv2.imshow("Original", img)
cv2.imshow("Undistorted (Full FOV)", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
