# PRODIGY_ML_INTERNSHIP

Task 1 : House Price Prediction using Linear Regression

1. Load the dataset for house price prediction using linear regression.
2. Select the required columns from the dataset.
3. Split the dataset into training and testing sets.
4. Train the linear regression model using the selected features.
5. Use the trained model to make predictions.
6. Calculate and report the mean squared error (MSE).
7. Calculate and report the R-squared (R²) value.
8. Determine and report the coefficients of the model.
9. Determine and report the intercept of the model.


Task 2 : K-means Clustering Algorithm for Grouping Retail Store Customers Based on Purchase History

1. Load the dataset containing the customers' purchase history.
2. Preprocess the data to ensure it's suitable for clustering (e.g., normalization).
3. Apply the K-means clustering algorithm to the data.
4. Determine the optimal number of clusters using methods like the elbow method.
5. Analyze and visualize the clustered groups to interpret customer segments.

Task 3 : SVM for classification for Dogs and Cats

1. Load the required libraries: Import necessary libraries like `numpy`, `pandas`, `matplotlib`, `sklearn`, `os`, `cv2`, etc.
2. Load the dataset: Access the Kaggle dataset for cats and dogs images.
3. Preprocess the data: 
   -> Resize images to a uniform size (e.g., 64x64).
   -> Convert images to grayscale if necessary.
   -> Normalize pixel values to a range of [0, 1].
   -> Label the images (0 for cats, 1 for dogs).
4. Split the dataset: Divide the data into training and testing sets.
5. Flatten the images: Convert the 2D image arrays into 1D feature vectors.
6. Train the SVM model: 
   -> Initialize the SVM classifier with appropriate parameters (e.g., kernel type).
   -> Fit the model using the training data and corresponding labels.
7. Make predictions: Use the trained SVM model to predict labels for the test set.
8. Evaluate the model: 
   -> Calculate the accuracy of the model on the test set.
   -> Generate a classification report including precision, recall, and F1-score.
9. Visualize the results: Plot sample images with their predicted labels to verify the model's performance.

Task 4 : Hand Gesture Classification

1. Load and preprocess the hand gesture dataset.
2. Extract relevant features from images or video frames.
3. Choose and define a suitable model architecture, such as CNN for image data.
4. Compile and train the model using the dataset.
5. Evaluate the model's performance using metrics like accuracy.
