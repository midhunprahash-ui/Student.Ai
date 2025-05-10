from google import genai

client = genai.Client(api_key="AIzaSyDNiBsbqkhe-nmze0uRwr80taQLCYfxpYo")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works in a few words",
)

