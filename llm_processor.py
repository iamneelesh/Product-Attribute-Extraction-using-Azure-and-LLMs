import time
import pandas as pd
from phi_model import get_phi_response


entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

def generate_attribute_prompt(entity_name):
    units = ', '.join(entity_unit_map[entity_name])
    return f"""
    You are given an image. Your task is to analyze this image properly and extract the {entity_name} of the product from this image.
    This value must be explicitly stated somewhere in the image itself.
    - You must analyze the image well.
    - The unit for the value must only be from these values: {units}
    - Do NOT abbreviate the unit, state it fully (e.g., 'gram' instead of 'g').
    - Only use the unit provided on the product, do NOT convert it.
    - If unclear, set quantity to 0 and unit to "uncertain".


    You MUST return only a JSON object in this format:
    {{
        "quantity": <number>,
        "unit": "<unit name>"
    }}

    Example:
    {{"quantity": 1.5, "unit": "kilogram"}}
    """

def get_product_attribute(image_url, entity_name):
    attribute_prompt = generate_attribute_prompt(entity_name)

    # start_time = time.time()
    # llm_response_openai = get_openai_response(attribute_prompt, image_url)
    # openai_time = time.time() - start_time

    start_time = time.time()
    llm_response_phi = get_phi_response(attribute_prompt, image_url)
    phi_time = time.time() - start_time

    return llm_response_phi, phi_time

def process_csv(train_csv_path, num_rows):
    df = pd.read_csv(train_csv_path)
    num_rows = min(num_rows, len(df))

    for i in range(num_rows):
        row = df.iloc[i]
        image_url = row['image_link']
        entity_name = row['entity_name']
        expected_value = row['entity_value']

        result_phi, phi_time = get_product_attribute(image_url, entity_name)

        print(f"\n{'='*50}")
        print(f"Image URL: {image_url}")
        print(f"Expected response: {expected_value}")
        # print(f"OpenAI answer: {result_openai}")
        # print(f"Time taken for OpenAI answer: {openai_time:.2f} seconds")
        print(f"Phi answer: {result_phi}")
        print(f"Time taken for Phi answer: {phi_time:.2f} seconds")

def main():
    train_csv = 'train.csv'
    num_rows_to_process = 25
    process_csv(train_csv, num_rows_to_process)

if __name__ == "__main__":
    main()