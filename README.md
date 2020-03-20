@ChatBot:

Dataset:
This should be in format like:
{"intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day", "Whats up"],
         "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"],
         "context_set": ""
        }
        ]
}

How to run:
1) Open terminal
2) pyhton model.py --data_path data/intents.json --model_path model/ --save_parameter parameter/parameter.pkl
(data_path, model_path, save_parameter are optional you can change it in the code in default option)

a) If you training for the first time :
python model.py --data_path data/intents.json --model_path model/ --save_parameter parameter/parameter.pkl
- it will create a model and parameter file
b) If files are already there then it will start conversation with Chat-Bot
