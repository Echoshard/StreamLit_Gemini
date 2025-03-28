## Gemini Streamlit

This is a Streamlit example that uses the Gemini API

This requires having a Gemini API key which can be aquired here https://aistudio.google.com/

### Supports

- Webscraping 
- youtube transcript scraping `Local only static IP's are blocked`
- File upload
- Vision
- Image generation `gemini-2.0-flash-exp-image-generation only` 
- System instructions
- Google search tool with sources
- Gemini 2.5

### Configuration

When running on streamlit cloud this can be configured for a secret key and your API key

```
gemini_key = "GEMINI_KEY"
SECRET_KEY = "SECRET"
```
You can also run it locally by setting the keys or on another service using environment variables just uncomment and comment what you need.


## Using the Secret Key

The secret key is designed to stop your app from being public entirely it can be activated by setting `requireKey` to true This will the require the user to put the correct param when navigating for example.

```
www.yourStreamLitAssistant.com/?secretkey=SECRET
```

## Local Running
- Download this repo and set your keys up
```
 pip install -r requirements.txt
```

```
 python -m streamlit run PATH:\\Gemini_StreamLit_Chat.py
```

# Hosting your App:
The fastest way to run this using streamlit using https://streamlit.io/gallery. 
Another site for doing this is https://render.com/

