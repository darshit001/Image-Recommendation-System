# Image Similarity Search using ResNet50

This project helps in finding similar marble tile images from a dataset based on an uploaded image using a pre-trained ResNet50 model and recommending the top 2 most similar images.
---

## Features

- **Feature Extraction**: Extracts image features using ResNet50 pre-trained on ImageNet.
- **Cosine Similarity**: Compares query image features with database image features.
- **Caching**: Saves extracted features in a `.pkl` file to avoid reprocessing.
- **Visualization**: Displays the query image and top-matching images with similarity scores.

## **Project Structure**

```
/image-similarity-search
│-- app.py               # Main Python script
│-- FIND THIS MARBLE/       # Folder containing the uploaded image
│   ├── onyx.jpg            # Image to find similar matches
│-- BACKUP MARBLE/          # Folder containing the dataset images
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
|-- requirements.txt        # all library
│-- README.md               # Project documentation
```

## **Running the Project**
1. **Clone the Repository**  
   ```sh
   git clone https://github.com/yourusername/image-similarity-search.git
   cd image-similarity-search
   ```

2. **Ensure Dependencies are Installed**  
   ```sh
   pip install -r requirements.txt
   ```

3. **Place Your Input Image**
   - Upload the folder which contain traning data (`BACKUP MARBLE/`)
   - Upload the folder which contain test data(`FIND THIS MARBLE/`)

5. **Run the Script**  
   ```sh
   python app.py
   ```

6. **View Results**  
   - The script will output the top 2 most similar images.
  


