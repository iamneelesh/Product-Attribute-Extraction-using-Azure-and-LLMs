import time
import os
import pandas as pd
import re
import json
from phi_model import get_phi_response
from llm_processor import entity_unit_map, generate_attribute_prompt

# Choose the model to use (either 'openai' or 'phi')
CHOSEN_MODEL = 'phi'


def get_product_attribute(image_url, entity_name):
    attribute_prompt = generate_attribute_prompt(entity_name)

    return get_phi_response(attribute_prompt, image_url)


def extract_quantity_and_unit(input_string):
    # Find the first occurrence of a JSON-like structure
    match = re.search(r'\{[^}]+\}', input_string)
    if match:
        try:
            # Parse the JSON-like structure
            data = json.loads(match.group())
            quantity = data.get('quantity')
            unit = data.get('unit')
            if isinstance(quantity, (int, float)) and isinstance(unit, str):
                return {
                    "quantity": quantity,
                    "unit": unit
                }
        except json.JSONDecodeError:
            pass

    return {}


def process_and_save(test_csv_path, output_csv_path):
    df = pd.read_csv(test_csv_path)

    # Check if the output file exists and load it if it does
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        last_processed_index = existing_df['index'].max()
        print(f"Resuming from index {last_processed_index + 1}")
        start_index = last_processed_index + 1
        results = existing_df.to_dict('records')
    else:
        start_index = 116933
        results = []

    for index, row in df.iloc[start_index:].iterrows():
        image_url = row['image_link']
        entity_name = row['entity_name']

        start_time = time.time()
        llm_response = get_product_attribute(image_url, entity_name)

        processed_output = extract_quantity_and_unit(llm_response)

        if isinstance(processed_output, dict) and 'quantity' in processed_output and 'unit' in processed_output:
            if processed_output['unit'].lower() == 'uncertain':
                prediction = ""
            else:
                prediction = f"{processed_output['quantity']} {processed_output['unit']}"
        else:
            prediction = ""

        results.append({'index': index, 'prediction': prediction})

        time_taken = time.time() - start_time
        print(f"Index: {index}, Image URL: {image_url}, Entity: {entity_name}, Prediction: {prediction}, Time taken: {time_taken:.2f} seconds")

        # Save progress after each processed row
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv_path, index=False, mode='w')  # 'w' mode to overwrite the file each time

    print(f"Results saved to {output_csv_path}")


def main():
    test_csv_path = 'test.csv'  # Update this path to your test.csv location
    output_csv_path = 'submission4.csv'
    process_and_save(test_csv_path, output_csv_path)


if __name__ == "__main__":
    main()