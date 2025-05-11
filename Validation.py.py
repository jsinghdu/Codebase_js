import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import os
# from dotenv import load_dotenv
import pandas as pd

# Set the API key as an environment variable
# load_dotenv()

api_key = os.getenv("ULLM_API_KEY")


# Define the API endpoint
url = url  # Replace with the actual API endpoint

if not api_key:
    raise ValueError("API key is not set. Please check your environment variables.")

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_taxonomy(csv_path):
    """
    Extracts taxonomy from a given CSV file and returns it as a dictionary.

    Args:
        csv_path (str): The path to the CSV file containing taxonomy data.

    Returns:
        dict: A dictionary where the keys are taxonomy names and values are their definitions.
    """
    try:
        taxonomy_df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        taxonomy_dict = {row["NAME"]: row["DEFINITION"] for _, row in taxonomy_df.iterrows()}
        return taxonomy_dict
    except Exception as e:
        print(f"Error loading taxonomy CSV: {e}")
        return {}

def categorize_course(course_name, short_description, taxonomy_dict, url, api_key):
    """
    Categorizes a course based on its name and description using LLM.

    Args:
        course_name (str): The name of the course.
        short_description (str): The short description of the course.
        taxonomy_dict (dict): The taxonomy dictionary to categorize against.
        url (str): The API URL for categorization.
        api_key (str): The API key for authentication.

    Returns:
        str: The suggested category for the course.
    """
    taxonomy_categories = ", ".join(taxonomy_dict.keys())  # List of category names

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a highly intelligent AI assistant specializing in course categorization.\n"
                f"The goal is to assign each course to one of the following predefined categories:\n\n"
                f"{taxonomy_categories}\n\n"
                f"Instructions: Read the course name and short description, compare with the predefined category names, "
                f"and return the exact category name that best matches the course content."
            )
        },
        {
            "role": "user",
            "content": (
                f"Course Name: {course_name}\n"
                f"Short Description: {short_description}\n\n"
                "Based on the above information, assign the most appropriate category."
            )
        }
    ]

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    try:
        response = requests.post(url, json={"messages": messages}, headers=headers)
        if response.status_code == 200:
            result = response.json()
            category = result["choices"][0]["message"]["content"].strip()
            return category
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error in sending request: {e}"

def compute_similarity(text1, text2):
    """
    Computes the similarity between two texts using TF-IDF vectorization and cosine similarity.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The similarity score as a percentage.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return round(similarity[0][0] * 100, 2)

def filter_products_by_application(product_df, application_value):
    """
    Filters products based on a given application value.

    Args:
        product_df (pd.DataFrame): The product data DataFrame.
        application_value (str): The application value to filter by.

    Returns:
        pd.DataFrame: A DataFrame of products matching the application value.
    """
    # Ensure 'Application' column exists in product DataFrame
    if 'Application' in product_df.columns:
        return product_df[product_df['Application'].str.strip() == application_value.strip()]
    else:
        print("Column 'Application' not found in product data.")
        return pd.DataFrame()

def suggest_best_product_line(filtered_products, api_key, url):
    """
    Suggests the best product line based on the filtered products using LLM.

    Args:
        filtered_products (pd.DataFrame): The filtered product DataFrame.
        api_key (str): The API key for authentication.
        url (str): The API URL for the suggestion request.

    Returns:
        str: The suggested product line.
    """
    if filtered_products.empty:
        return None

    product_lines = filtered_products['Product Line'].tolist()
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a product expert. Based on the following available product lines and their definitions, "
                f"provide the best possible product line.\n\n"
                f"Available Product Lines: {product_lines}"
            )
        },
        {
            "role": "user",
            "content": (
                "Based on these product lines, suggest the best possible product line."
            )
        }
    ]

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    try:
        response = requests.post(url, json={"messages": messages}, headers=headers)
        if response.status_code == 200:
            result = response.json()
            suggested_product_line = result["choices"][0]["message"]["content"].strip()
            return suggested_product_line
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error in sending request: {e}"

def main():
    """
    Main function to load data, categorize courses, compute similarity, and suggest product lines.
    """
    # Load taxonomy categories
    taxonomy_csv_path = "taxonomy_LLM.csv"
    taxonomy_dict = extract_taxonomy(taxonomy_csv_path)

    if taxonomy_dict:
        input_csv_path = "sn_lxp_course.csv"
        product_csv_path = "Data/product_data.csv"  # Path to your product data CSV file
        output_csv_path = "categorized_courses_with_validation.csv"
        
        # Load the course data from CSV
        try:
            df = pd.read_csv(input_csv_path, encoding='ISO-8859-1', nrows=3000)
        except Exception as e:
            print(f"Error loading course data CSV: {e}")
            df = pd.DataFrame()  # Create an empty DataFrame to prevent further errors
        
        # Add new columns for categories, similarity score, and derived product line
        df["LLM_Category"] = ""
        df["Similarity_Score (%)"] = 0.0
        df["derived_product_line_LLM_derived"] = ""

        # Load product data
        try:
            product_df = pd.read_csv(product_csv_path, encoding='ISO-8859-1')
        except Exception as e:
            print(f"Error loading product data CSV: {e}")
            product_df = pd.DataFrame()  # Create an empty DataFrame to prevent further errors

        # Iterate over each course and categorize
        for index, row in df.iterrows():
            course_name = row["name"]
            short_description = row["short_description"]
            application_value = row["Application"]
            combined_text = f"{course_name} {short_description}"

            # Categorize using LLM
            category = categorize_course(course_name, short_description, taxonomy_dict, url, api_key)
            df.at[index, "LLM_Category"] = category

            # Compute similarity score
            if category and category not in ["Error", "No category found in response"]:
                similarity_score = compute_similarity(combined_text, category)
                df.at[index, "Similarity_Score (%)"] = similarity_score
            else:
                df.at[index, "Similarity_Score (%)"] = 0.0

            # Perform second LLM call to suggest the best product line based on application
            if category:
                filtered_products = filter_products_by_application(product_df, application_value)
                suggested_product_line = suggest_best_product_line(filtered_products, api_key, url)
                df.at[index, "derived_product_line_LLM_derived"] = suggested_product_line

        # Save the updated DataFrame with LLM category, similarity score, and derived product line
        df.to_csv(output_csv_path, index=False)
        print(f"Categorized and validated data saved to {output_csv_path}")
    else:
        print("Failed to extract taxonomy data.")

if __name__ == "__main__":
    main()
