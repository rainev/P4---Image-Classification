# Image Classifier Dashboard
- Isaac A. Dicayanan
- Raine Miguel V. Villaver

## Installations for the Project
### Step 1: Clone / Download the Project
### Step 2: Create Virtual Environment
```bash
python -m venv .venv
```
### Step 3: Activate Virtual Environment
```bash
.venv\Scripts\activate
```
### Step 4: Install Dependencies
To install everything:
```bash
pip install PyQt6==6.10.0 pyqtgraph==0.14.0 grpcio==1.62.2 grpcio-tools==1.62.2 protobuf==4.25.3 torch==2.2.1 torchvision==0.17.1 pillow==11.3.0 numpy==1.26.4
```
Below are the dependencies above 


Dashboard app dependencies:
```bash
pip install PyQt6==6.10.0
pip install pyqtgraph==0.14.0
pip install grpcio==1.62.2
pip install grpcio-tools==1.62.2
pip install protobuf==4.25.3
```
Training app dependencies:
```bash
pip install torch==2.2.1
pip install torchvision==0.17.1
pip install pillow==11.3.0
pip install numpy==1.26.4
```

## Running the App
### Method 1: With Python Scripts
**Start the Dashboard**
1. Open a terminal from the project directory
2. Activate the virtual environment
3. Navigate to the dashboard folder
```bash
cd dashboard
```
4. Run the dashboard
```bash
python main.py
```

**Start the Training**
1. Open a new command prompt
2. Activate the virtual environment
3. Navigate to the training folder:
```bash
cd training
```
4. Run the training:
```bash
python train.py
```


