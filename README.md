# Sheet-counting-Application

The Sheet Counter Project is an innovative computer vision application designed to accurately count the number of sheets in a stack by leveraging a powerful combination of **OpenCV** for advanced image processing, **SymPy** for robust symbolic mathematics, and **Streamlit** for a sleek and interactive web interface. This project showcases custom implementations of Canny edge detection and Hough Line Transform, ensuring precise edge and line detection, while also integrating NumPy for efficient numerical operations. The intuitive web interface allows users to effortlessly upload images and view results in real-time.


## How to Run the Sheet Counter code:
Follow these steps to set up and run the sheet counter:

### 1. Create a New Folder:
Open VS Code and create a new folder for the project.
### 2. Create and activate a virtual environment.
Within the new folder, create and activate a virtual environment.
### 3. Upload Project Files:
Upload the following files into the folder:
* sheet_counter.py
* requirements.txt
### 4. Install Required Packages:
Install all required packages by running the following command:
**pip install -r requirements.txt**
### 5. Run the Streamlit Application:
Open a terminal and run the following command to start the Streamlit chatbot interface:
**streamlit run query_ui.py**
### 6. Upload an Image
* A web browser will open displaying the Streamlit **Sheet Counter** interface.
* Upload an image of a sheet stack in JPEG, JPG, or PNG format.
### 7. View Results
The application will display the **edge-detected image** and the **estimated number of sheets** in the stack.
