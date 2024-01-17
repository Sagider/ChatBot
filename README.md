This a simple chatbot that uses tensorflow to make neural networks to analyse psatterns in the given dataset(intents) to give responses to the text input from a user.
This project taught me a little bit about natural language processing
Requirements:
1. Preferrably run it in a virtual environment with tensorflow with Cuda (or metal for mac) plugins installed
2. Python ver 3.11
3. Tensorflow ver 2.15.0
4. Pillow 9.5.0
5. tflearn 0.5.0, in "recurrent.py" (python3.11/site-packages/tflearn/layers/recurrent.py) (The path may be different for you), in line 17, change 'is_sequence' to 'is_sequence_or_composite'
6. nltk 3.8.1
7. numpy 1.26.3
8. also install the 'punkt' stemmer from nltk (you can use pip install, for further details, visit https://www.nltk.org)

main.py is based on the program by Tech With Tim, testlearn.py is an experiment where I am trying to train the model using super_intents.json which is more detailed, as of now, it is faulty and only has an accuracy of about 20%
