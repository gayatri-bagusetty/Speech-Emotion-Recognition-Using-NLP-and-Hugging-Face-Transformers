**DATASET:**

Dataset is downloaded from the below link.Only 10 audio file downloaded.

Link - https://www.voiptroubleshooter.com/open_speech/american.html

**CODE DETAILS:**

_Install Libraries:_
!apt-get install portaudio19-dev – Install PortAudio.
!pip install pyaudio SpeechRecognition nltk transformers.

_Import Libraries:_
Import essential libraries: os, nltk, speech_recognition, transformers, etc.

_Download NLTK Data:_
nltk.download('stopwords') and nltk.download('punkt') for text processing.

_Load Emotion Model:_
Load Hugging Face DistilBERT emotion model using pipeline().

_Preprocess Text:_
Convert text to lowercase, remove punctuation, tokenize, and remove stopwords (except negations).

_Detect Emotion:_
Clean input text and classify emotion using the loaded model.

_Speech-to-Text Conversion:_
Use speech_recognition to convert audio files to text with timestamps.

_Process Audio Dataset:_
Loop through audio files in a folder, convert each to text, and detect emotion.

_Run in Google Colab:_
Mount Google Drive to access dataset and run the process_audio_dataset() function.
