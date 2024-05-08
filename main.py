import cv2
import pytesseract
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from google.colab import files

def load_image(image_path):
    # Load the image from the specified path
    image = cv2.imread(image_path)
    return image

def extract_text(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform thresholding to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the binary image
    binary = cv2.bitwise_not(binary)

    # Perform OCR to extract text
    text = pytesseract.image_to_string(binary)

    return text

def generate_summary(text, num_sentences=3):
    # Split the text into sentences
    sentences = text.split('.')

    # Calculate TF-IDF scores for the sentences
    tfidf = TfidfVectorizer().fit_transform(sentences)
    scores = np.sum(tfidf, axis=1)

    # Get the indices of top-scoring sentences
    top_sentences = np.argsort(scores, axis=0)[-num_sentences:]
    top_sentences = sorted(top_sentences)

    # Convert top_sentences to regular integers
    top_sentences = [int(i[0]) for i in top_sentences]

    # Generate summary by concatenating top-scoring sentences
    summary = '. '.join(sentences[i] for i in top_sentences)

    return summary

def main():
    # Upload an image file
    print("Please upload an image file:")
    uploaded_file = files.upload()

    # Get the file name of the uploaded image
    image_path = list(uploaded_file.keys())[0]

    # Load the image
    image = load_image(image_path)

    # Extract text from the image
    text = extract_text(image)

    # Generate summary of the extracted text
    summary = generate_summary(text)

    # Print the original text and its summary
    print("\nOriginal Text:")
    print(text)
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    main()
