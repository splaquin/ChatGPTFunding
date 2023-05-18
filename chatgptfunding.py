import openai
import pandas as pd
import time 

# Record the start time
start_time = time.time()


def generate_chatgpt_response(text_block):
    # Your OpenAI API key
    openai.api_key = 'OPEN AI API KEY'

    # Prepare the instruction
    instruction = f"Analyze the following text for any funding opportunities for researchers: {text_block}"

    while True:
        try:
            # Make the API call with the chat-based model
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that identifies funding opportunities for researchers in text."
                    },
                    {
                        "role": "user",
                        "content": instruction
                    }
                ],
                temperature=0.5,  # Adjust the temperature to control randomness
                max_tokens=50     # Limit the response length
            )

            # Extract the assistant's reply
            funding_opportunities = response['choices'][0]['message']['content']
            return funding_opportunities.strip()
        except openai.error.RateLimitError:
            print("Rate limit hit. Waiting for 60 seconds before retrying.")
            time.sleep(60)

def generate_funding_opportunity_decision(text_block):
    # Prepare the instruction
    instruction = f"Does the following text specifically mention a funding opportunity for a researcher? {text_block}"

    while True:
        try:
            # Make the API call with the chat-based model
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that identifies funding opportunities for researchers in text. Respond only with 'Yes', 'No', or 'Undetermined'"
                    },
                    {
                        "role": "user",
                        "content": instruction
                    }
                ],
                temperature=0.0,  # Adjust the temperature to control randomness
                max_tokens=1     # Limit the response length
            )

            # Extract the assistant's reply
            decision = response['choices'][0]['message']['content']
            return decision.strip()
        except openai.error.RateLimitError:
            print("Rate limit hit. Waiting for 60 seconds before retrying.")
            time.sleep(60)

# Read the CSV file
df = pd.read_csv('input.csv')  # Replace with your CSV file path

# Analyze each row for funding opportunities
df['Response'] = df['Text'].apply(generate_chatgpt_response)  # Replace 'Text' with your text column name
df['Funding Opportunity'] = df['Text'].apply(generate_funding_opportunity_decision)

# Write the results back to the CSV file
df.to_csv('output.csv', index=False)  # Replace with your output CSV file path

# Record the end time
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

print(f"Script duration: {duration} seconds")
