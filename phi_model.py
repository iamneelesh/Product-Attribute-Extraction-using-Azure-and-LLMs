from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage, TextContentItem, ImageContentItem, ImageUrl, ImageDetailLevel

# API Key and client setup
api_key = 'your-api-key'
client = ChatCompletionsClient(
    endpoint='https://Phi-3-5-vision-instruct-esidt.eastus.models.ai.azure.com',
    credential=AzureKeyCredential(api_key)
)

def get_phi_response(prompt, image_url):
    try:
        response = client.complete(
            messages=[
                UserMessage(
                    content=[
                        TextContentItem(text=prompt),
                        ImageContentItem(image_url=ImageUrl(url=image_url, detail=ImageDetailLevel.HIGH)),
                    ],
                ),
            ],
            temperature=0.1,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"