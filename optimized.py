import requests
import json
import os
from dotenv import load_dotenv
import pandas as pd
import os
os.environ['CURL_CA_BUNDLE'] = ''
import requests
from huggingface_hub import configure_http_backend
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)
# Set the API key as an environment variable
load_dotenv()

api_key = os.getenv("ULLM_API_KEY")

# Define the API endpoint
url = "https://apidev.servicenow.com/ullm_dev/ullm-router/score"  # Replace with the actual API endpoint

if not api_key:
    raise ValueError("API key is not set. Please check your environment variables.")

# Categories to classify text into
categories = ["Customer Support", "IT Operations", "HR Management", "Marketing", "Sales"]

def categorize_text(text):
    """
    Categorizes the given text using the LLM API.
    
    :param text: The text to categorize.
    :return: The category name or error details.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant tasked with categorizing text into one of the following categories: "
                f"{', '.join(categories)}. "
                "Respond with the category name only. Examples: "
                "1. 'Handle customer complaints and support queries' -> Customer Support "
                "2. 'Automated monitoring of IT systems' -> IT Operations "
                "3. 'Streamlining employee onboarding' -> HR Management "
                "4. 'Social media ads and campaigns' -> Marketing "
                "5. 'Analyzing customer data to boost sales' -> Sales"
            )
        },
        {
            "role": "user",
            "content": f"Text: {text}"
        }
    ]

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    # Make the API call
    response = requests.post(url, json={"messages": messages}, headers=headers)

    # Print raw response for debugging
    print(f"Raw response for text: '{text}':\n{response.text}\n")

    # Process the response
    if response.status_code == 200:
        try:
            result = response.json()
            return result.get("content", "No category returned").strip()
        except KeyError:
            return "Invalid response structure."
        except ValueError:
            return "Response parsing error."
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example texts to categorize
texts = [
    "ServiceNow helps streamline HR processes to improve employee onboarding.",
    "Marketing campaigns on social media boost customer engagement.",
    "IT operations are optimized using ServiceNow's automation tools.",
    "Customer support queries are managed efficiently with ServiceNow's CSM.",
    "Sales data analysis reveals trends to enhance revenue."
]

# Categorize each text
for text in texts:
    category = categorize_text(text)
    print(f"Text: {text}\nCategory: {category}\n")



import chardet

# Detect encoding
with open("taxonomy_LLM.csv", "rb") as f:
    result = chardet.detect(f.read())
    encoding = result["encoding"]
    print(f"Detected encoding: {encoding}")

# Load the file with the detected encoding





import pandas as pd
import asyncio
import aiohttp
import json

# Function to extract taxonomy categories from CSV
def extract_taxonomy(csv_path):
    try:
        taxonomy_df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        taxonomy_dict = {row["NAME"]: row["DEFINITION"] for _, row in taxonomy_df.iterrows()}
        return taxonomy_dict
    except Exception as e:
        print(f"Error extracting taxonomy: {e}")
        return {}

# Asynchronous function to categorize a course
async def categorize_course(course_name, short_description, taxonomy_dict, session):
    taxonomy_categories = ", ".join(taxonomy_dict.keys())
    taxonomy_context = "\n".join([f"{NAME}: {DEFINITION}" for NAME, DEFINITION in taxonomy_dict.items()])
    
    messages = [
        {
            "role": "system",
            "content": (
                f"You are an AI assistant that categorizes courses into one of the following categories: {taxonomy_categories}.\n"
                f"Here are the category definitions:\n{taxonomy_context}\n"
                f"Use the course name and description to determine the best fit."
            )
        },
        {
            "role": "user",
            "content": (
                f"Course Name: {course_name}\n"
                f"Short Description: {short_description}\n"
                "Provide the best category for this course:"
            )
        }
    ]

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    

    try:
        async with session.post(url, json={"messages": messages}, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("content", "No category returned")
            else:
                return f"Error: {response.status}, {await response.text()}"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to process the CSV and categorize courses asynchronously
async def process_courses(input_csv_path, taxonomy_dict, output_csv_path, batch_size=20):
    df = pd.read_csv(input_csv_path, encoding='ISO-8859-1', nrows=100)  # Load first 100 rows
    df["Category"] = ""

    # Prepare async session
    async with aiohttp.ClientSession() as session:
        # Initialize a list to store all categorized courses
        categorized_courses = []
        total_processed = 0  # Variable to track the total number of processed records
        
        # Iterate over rows and prepare categorization tasks in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]  # Get the next batch of rows
            tasks = []
            
            # Create tasks for the current batch
            for index, row in batch.iterrows():
                course_name = row["name"]
                short_description = row["short_description"]
                tasks.append(categorize_course(course_name, short_description, taxonomy_dict, session))
            
            # Run all tasks for the batch concurrently
            categories = await asyncio.gather(*tasks)
            
            # Update the DataFrame with the categories for this batch
            df.loc[i:i + batch_size - 1, "Category"] = categories

            # Save the current batch to the output CSV
            df.iloc[i:i + batch_size].to_csv(output_csv_path, mode='a', header=(i == 0), index=False)
            
            # Update total processed records count
            total_processed += len(batch)

            # Print the progress
            for j, category in enumerate(categories):
                current_row_index = i + j
                print(f"Processing row {current_row_index + 1}: {df.loc[current_row_index, 'name']}, Category: {category}")

            # Print batch progress
            print(f"Processed batch {i // batch_size + 1}, records {i + 1} to {min(i + batch_size, len(df))}. Total records processed: {total_processed}")

    print(f"Categorized data saved to {output_csv_path}")
    print(f"Total records processed: {total_processed}")

# Main execution
def main():
    taxonomy_csv_path = "taxonomy_LLM.csv"  # Path to your taxonomy CSV file
    input_csv_path = "sn_lxp_course.csv"  # Path to your input courses CSV file
    output_csv_path = "categorized_courses.csv"  # Path to save the output CSV file

    taxonomy_dict = extract_taxonomy(taxonomy_csv_path)

    if taxonomy_dict:
        # Run the async process
        asyncio.run(process_courses(input_csv_path, taxonomy_dict, output_csv_path))
    else:
        print("Failed to extract taxonomy data.")

if __name__ == "__main__":
    main()
