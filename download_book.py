import requests

base_url = "https://e-teaching.phonics-smart.edu.vn/storage/Tieng_Anh_2_Phonics_Smart/Flipbook/Activity_book/files/mobile/"

for page in range(1, 61):
    url = f"{base_url}{page}.jpg?220712101452"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(f"./book/page{page}.jpg", "wb") as file:
            file.write(response.content)
            print(f"Page {page} saved successfully.")
    else:
        print(f"Failed to download page {page}.")
