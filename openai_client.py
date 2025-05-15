import os
import google.generativeai as genai

# Initialize the Gemini API with your key
api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyDNiBsbqkhe-nmze0uRwr80taQLCYfxpYo')
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-pro')

def generate_response(prompt):
    """
    Generate a response using the Gemini model
    Args:
        prompt (str): The prompt to send to the model
    Returns:
        str: The model's response
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

