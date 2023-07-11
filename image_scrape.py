# Import the required modules
import requests
import json
import os

# Define the API key and the search engine ID
api_key = "AIzaSyDI3E8qZa6eqmaDOLhlYdpjO-u9NZneziI"
search_engine_id = "f69dd879caadf4092"

# Define the text query and the number of images to download
text_query = "hello"
num_images = 10

# Create a directory to store the images
image_dir = os.path.join(os.getcwd(), text_query)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Construct the URL for the Google Custom Search API
url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={text_query}&searchType=image&num={num_images}"

# Make a request to the API and get the response as a JSON object
response = requests.get(url)
response_json = response.json()

# Loop through the items in the response and download each image
for i, item in enumerate(response_json["items"]):
    # Get the image link and the file name
    image_link = item["link"]
    file_name = os.path.join(image_dir, f"{text_query}_{i}.jpg")

    # Download the image and save it to the file
    image = requests.get(image_link).content
    with open(file_name, "wb") as f:
        f.write(image)

    # Print a message to indicate the progress
    print(f"Downloaded {file_name} from {image_link}")
