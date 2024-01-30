from dotenv import load_dotenv

load_dotenv()
import os
from openai import AzureOpenAI


# must have below env setting:
# AZURE_OPENAI_API_KEY
# OPENAI_API_VERSION
# AZURE_OPENAI_ENDPOINT
client = AzureOpenAI()

# Make sure the environment variable OPENAI_API_KEY is set.

# Call the openai ChatCompletion endpoint, with th ChatGPT model
response = client.chat.completions.create(model="gpt-35-turbo-tdu",
messages=[
      {"role": "user", "content": "Hello World!"}
  ])

# Extract the response
print(response.choices[0].message.content)
