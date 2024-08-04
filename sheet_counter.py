import cv2
import numpy as np
import streamlit as st
import sympy as sp

# Define the variable b for solving equations later
sp.var('b,x')

def canny_detector(img, weak_th=None, strong_th=None):
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction step
    img = cv2.GaussianBlur(img, (5, 5), 1.4)

    # Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    # Conversion of Cartesian coordinates to polar
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Setting the minimum and maximum thresholds for double thresholding
    mag_max = np.max(mag)
    if not weak_th: weak_th = mag_max * 0.1
    if not strong_th: strong_th = mag_max * 0.5

    # Getting the dimensions of the input image
    height, width = img.shape

    # Looping through every pixel of the grayscale image
    for i_x in range(width):
        for i_y in range(height):
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # Selecting the neighbours of the target pixel according to the gradient direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y - 1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y - 1
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = i_x - 1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # Non-maximum suppression step
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x] = 0
                    continue
            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x] = 0

    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)
    ids = np.zeros_like(img)

    # Double thresholding step
    for i_x in range(width):
        for i_y in range(height):
            grad_mag = mag[i_y, i_x]
            if grad_mag < weak_th:
                mag[i_y, i_x] = 0
            elif strong_th > grad_mag >= weak_th:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2

    return mag

def hough_line_transform(edges):
    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=20)
    return lines

def filter_and_count_sheets(lines, img_shape):
    if lines is None:
        return 0
    
    mid_xs = []
    img = np.zeros(img_shape)

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1:  # Avoid division by zero
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = sp.solve(sp.Eq(slope * x1 + b, y1))
        if not intercept:
            continue
        intercept = intercept[0]
        mid_x_solution = sp.solve(slope * x + intercept - 200)
        if not mid_x_solution:
            continue
        mid_x = float(mid_x_solution[0])
        mid_xs.append(mid_x)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    if not mid_xs:
        return 0

    mid_xs = np.sort(np.array(mid_xs))
    distances = mid_xs[1:] - mid_xs[:-1]
    median = np.median(distances)

    def calc_mean_sq_error(arr):
        distances = arr[1:] - arr[:-1]
        return np.mean((distances - median) ** 2)

    @np.vectorize
    def decide_what_to_remove(i):
        arr = mid_xs[i - 1:i + 3]
        if len(arr) < 4:  # Ensure arr has enough elements
            return i
        mse1 = calc_mean_sq_error(np.delete(arr, 1))
        mse2 = calc_mean_sq_error(np.delete(arr, 2))
        return i if mse1 < mse2 else i + 1

    problematic_distances = np.where(np.abs(distances - median) > 15)[0]
    if len(problematic_distances) > 0:
        to_delete = decide_what_to_remove(problematic_distances)
        mid_xs = np.delete(mid_xs, to_delete, axis=0)
    
    return len(mid_xs)

def preprocess_image(file):
    # Convert the uploaded file to a numpy array
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    # Decode the numpy array into an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Apply Gaussian blur multiple times
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for _ in range(3):
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply the custom Canny edge detector
    edges = cv2.Canny(gray, 50, 150)

    return edges

def main():
    st.title("Sheet Counter")
    st.write("Upload an image of the sheet stack to get the count of sheets.")
    
    uploaded_file = st.file_uploader("Choose an image....", type=["jpeg", "jpg", "png"])
    if uploaded_file is not None:
        # Preprocess image
        edges = preprocess_image(uploaded_file)
        
        # Convert edges to RGB for displaying
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Display the edge-detected image
        st.image(edges_rgb, caption="Edge Detected Image", use_column_width=True)
        
        # Apply Hough Line Transform and count sheets
        lines = hough_line_transform(edges)
        if lines is not None:
            sheet_count = filter_and_count_sheets(lines, edges.shape)
        else:
            sheet_count = 0
        
        st.write(f"Number of sheets in the stack: {sheet_count}")

if __name__ == "__main__":
    main()



