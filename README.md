# Snackopia - Smart Billing System

## Introduction

**Evolution of Retail Automation –**  Traditional retail checkout systems relied on barcode scanning and manual billing. With computer vision and deep learning advancements, automated object detection has revolutionized the shopping experience.
 
**Object Detection –** AI-powered systems such as YOLOv8 allow for immediate product identification, removing the requirement for barcode scanning and decreasing checkout time in retail stores.
 
**Need for Efficient Billing Systems –** Long queues, manual errors, and slow processing highlight the need for a fast, reliable, and fully automated billing solution, ensuring a seamless shopping experience.

## Problem Statement
The traditional retail billing process relies on manual barcode scanning, leading to inefficiencies such as long checkout lines, human errors, and dependency on intact barcodes. These limitations result in delays and inconvenience for both customers and cashiers.

To address these challenges, Snack-O-Pia introduces an AI-powered smart billing system using YOLOv8 for object detection. By identifying products through a live camera feed and retrieving price details from a database, the system automates the billing process, eliminating the need for barcode scanning and enhancing efficiency in retail transactions.

## Workflow
The system follows this data stream process: 
1. Users upload a recorded video via the Web Interface
2. Yolov8 analyzes frames to detect objects. 
3. Detected objects are connected to database entries. 
4. The bill is consistently updated and displayed on the user interface.

<img src="https://media-hosting.imagekit.io/42978bf1ec87472a/Screenshot_13.jpg?Expires=1839997042&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=TMJe2su8VAn~ICvPNjTjsY1EnXe05GlZPIJ9gF6nQTlg9-8ruaJZUxyAwN0pz0DMwbxe6PmO9AolEkPY0RbMdDoupmgd76rNrp2JhpdsTo6pghaxZOhvi54fn0bbDsBQrxwb-PctjgOSEx9M72tm0mWTJmmA3KWBL41lsIds5~evli4EPiHrBDVladsbecrSvm3TDBCo7nWL6OWDQyCzHKpvmyhMn9~tNwJp2PLqopho~kfp4biHvY~Cl~N80yTRSe7elq98B6LeBdAZbnaGKvLJTbJ0QP2TqjfoX34hOenw7yPcSlOG5xrsOtS2Hcaxtlw8xFKQ4nGyvK~4bL6Rcw__" width="400"/>

## Implementation

### <ins>Data Collection & Preprocessing</ins>

**Captured images of 8 snack items using a webcam:**
- Each team member captured images of two different snack items from various angles.
- Diverse lighting conditions and backgrounds were ensured to improve model generalization.
- A total of 2307  images were collected to enhance the dataset's robustness.

**Used Roboflow for dataset augmentation & annotation:**
- Annotation: Manually labeled each snack item using Roboflow’s bounding box tool.
- Augmentation: Applied rotation, brightness adjustments, and flipping to increase dataset diversity. 
- Exported dataset in YOLOv8-compatible format for training.

**Splitting the dataset into training, validation, and testing sets:**
- Training Set: 70% of images for model learning.
- Validation Set: 20% of images for tuning hyperparameters. 
- Testing Set: 10% of images for evaluating model performance.
- Ensured a balanced distribution of each snack item in all sets.

### <ins>Model Training with YOLOv8 on Google Colab</ins>

**Utilized Google Colab for Training the YOLOv8 Model:** Leveraged cloud-based GPU resources to efficiently train YOLOv8 using the dataset from Roboflow.

**Performed Training for 300 Epochs with a Dataset from Roboflow:** Model trained for 300 iterations to ensure higher accuracy and optimal object detection capability.

**Optimized Training by Adjusting Epochs, Batch Size, and Learning Rate:** Tweaked hyperparameters like batch size, learning rate, and epochs for enhanced model performance.

**Generated Confusion Matrix to Analyze Detection Accuracy:** Created a confusion matrix to visualize misclassifications and evaluate precision and recall rates

### <ins>Web Interface & User Interaction</ins>

**Flask-Based Web Application:** Flask was chosen as the backend framework to handle communication between the YOLOv8 model and the front-end.The detection results are processed after being uploaded. The Flask app hosts a local web server where users can view detected products and their prices.

**Displaying Detected Products & Prices:** When the YOLOv8 model detects an item, its name and price are retrieved from the database (MySQL).

**Front-End Design Using HTML, CSS:** HTML structures the web page layout.CSS & Bootstrap are used to create a clean, responsive design.

### <ins>Database & Billing System </ins>

**Used MySQL for product storage & billing calculations.** The database stores product names, IDs, and prices, ensuring accurate retrieval and efficient calculation of total billing amounts. 

**The total cost is automatically updated when new items are added when the threshold is passed.** If there are multiple instances of an item, the systemdynamically updates the total bill.

### <ins>Final Integration & Testing</ins>

- Integrated object detection, web interface, and database to ensure smooth real-time interaction and accurate billing.
- Checked database reliability by testing multiple transactions to confirm proper item storage and retrieval.
- Ensured seamless synchronization between the YOLOv8 model and the billing system to prevent miscalculations.
- Checked the integration of OpenCV for smooth video processing without lag during object recognition.
- Validated UI functionality to display detected products and total cost dynamically without manual updates.

## Result

**1.Successful Integration:** The YOLOv8 model successfully detected snack items in
 real time.The Flask-based web interface efficiently displayed detected items and
 their prices.
 
**2. Accuracy and Efficiency:** Model trained on 2307 images achieved high accuracy in object detection. The confusion matrix showed minimal misclassification, improving billing precision.

**3. Smooth User Experience:** The real-time system reduced checkout time significantly compared to manual barcode scanning. The total bill updated dynamically as new items were detected.

**4. Challenges Identified:** Detection accuracy was affected by poor lighting conditions.Overlapping objects sometimes led to misclassification.

## Conclusion

The Snack-O-Pia project successfully demonstrates the potential of AI-driven automation in retail billing. By leveraging YOLOv8 for real-time object detection, Roboflow for dataset preparation, and Flask for deployment, the system effectively eliminates the need for barcode scanning, streamlining the checkout process. The integration of OpenCV for real-time image processing and DBMS for automated invoicing ensures accurate and efficient billing.

This project addresses key retail challenges such as long checkout queues, manual errors, and inefficiencies in billing systems. Our implementation showcases the feasibility of AI-powered automation in retail environments, offering a scalable and cost-effective alternative to traditional billing methods.

Despite its success, challenges like lighting variations, overlapping object detections, and dataset limitations were observed. However, these can be mitigated with further optimizations in model training and dataset diversity.

## Future Scope
- **Enhanced Object Recognition:** Improving the model with a larger dataset covering a broader range of products.Implementing transfer learning to refine detection accuracy for varied retail environments.
- **Integration with Payment Gateways:** Adding an automatic payment system using UPI, credit cards, or mobile wallets for seamless transactions.
- **Multi-Camera Setup:** Utilizing multiple cameras to expand detection coverage and eliminate blind spots.
- **Edge AI and IoT Implementation:** Deploying the model on low-power edge devices (e.g., Raspberry Pi, Jetson Nano) for a cost-effective, detection system without reliance on cloud computing.
- **Smart Inventory Management:** Connecting with inventory databases to update stock levels automatically as purchases are made
