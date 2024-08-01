from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
#from tensorflow.keras.models import load_model
from PIL import Image
import io
import google.generativeai as genai
#from gemini import Gemini

app = Flask(__name__,static_url_path='/static')


genai.configure(api_key="your api key")

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 0,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

gmodel = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)



# Load the model
model = keras.models.load_model('model3.keras')
class_dict = np.load("80_class_names.npy", allow_pickle=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['file']
    
    # Read the image file
    img = Image.open(io.BytesIO(file.read()))
    
    # Preprocess the image
    # (add any preprocessing steps here if needed)

    # Make prediction
    img_array = np.array(img.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_class_name = class_dict[predicted_class]
    convo = gmodel.start_chat(history=[])

    convo.send_message("Give me the accurate medicinal properties, adverse affects, cultivation techniques and geographical distribution for this plant: "+predicted_class_name+" with the respective headings. please use trustworthy sources")
    gen_response = convo.last.text
    print(gen_response)
    if isinstance(gen_response, str):
      parsed_response = {}
      current_section=None
      for line in gen_response.split('\n'):
          if line.strip() == '':
              continue
          if '**' in line:
            current_section = line.strip('**')
            parsed_response[current_section] = []
          else:
            parsed_response[current_section].append(line.strip())
      gen_response = parsed_response

    #print(gen_response)
    return render_template('result.html', predicted_class=predicted_class_name,generative_response= gen_response)




if __name__ == '__main__':
    app.run(debug=True)
