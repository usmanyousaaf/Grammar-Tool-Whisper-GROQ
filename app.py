
import os
from flask import Flask, request, render_template, jsonify
from groq import Groq

# Set up the Groq client
os.environ["GROQ_API_KEY"] = "your_api_key"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

app = Flask(__name__)

# Route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the audio upload and transcription
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_data = request.files['audio_data']
    selected_language = request.form['language']

    try:
        # Transcribe the audio based on the selected language
        transcription = client.audio.transcriptions.create(
            file=(audio_data.filename, audio_data.read()),
            model="whisper-large-v3",
            prompt="Transcribe the audio accurately based on the selected language.",
            response_format="text",
            language=selected_language,
        )

        return jsonify({'transcription': transcription})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to check grammar and vocabulary of the transcription
@app.route('/check_grammar', methods=['POST'])
def check_grammar():
    transcription = request.form.get('transcription')
    selected_language = request.form.get('language')

    if not transcription or not selected_language:
        return jsonify({'error': 'Missing transcription or language selection'}), 400

    try:
        # Grammar check based on the selected language
        # grammar_prompt = f"Briefly check the grammar of the following text in {selected_language}: {transcription}"
        grammar_prompt =(
            f"Briefly check the grammar of the following text in {selected_language}: {transcription}. "
            "Identify any word that does not belong to the selected language and flag it. Based on the number of incorrect words  also check the grammer deeply and carefully "
            "Provide a score from 1 to 10 based on the grammar accuracy, reducing points for incorrect words and make sure to output the score on a new line after two line break like ""SCORE=""."
        )


        grammar_check_response  =client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=[{"role": "user", "content": grammar_prompt}]
        )
        grammar_feedback = grammar_check_response.choices[0].message.content.strip()

        # Vocabulary check
        # vocabulary_prompt = f"Briefly check the vocabulary accuracy of the following text in {selected_language}: {transcription}"
       
        # Updated vocabulary prompt
        vocabulary_prompt = (
            f"Check the vocabulary accuracy of the following text in {selected_language}: {transcription}. "
            "Identify any word that does not belong to the selected language and flag it. Based on the number of incorrect words also check the grammer deeply and carefully "
            "provide a score from 1 to 10,based on the vocabulary accuracy reducing points for incorrect words and make sure to output the score on a new line after two line break like ""SCORE=""."
        )

        vocabulary_check_response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": vocabulary_prompt}]
        )
        vocabulary_feedback = vocabulary_check_response.choices[0].message.content.strip()

        # Return feedback and scores
        grammar_score = calculate_score(grammar_feedback)
        vocabulary_score = calculate_score(vocabulary_feedback)

        return jsonify({
            'grammar_feedback': grammar_feedback,
            'vocabulary_feedback': vocabulary_feedback,
            'grammar_score': grammar_score,
            'vocabulary_score': vocabulary_score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

def calculate_score(feedback):
    """ 
    Calculate score based on feedback content. 
    This function searches for the keyword 'SCORE=' or similar variations 
    (SCORE:, score:, etc.) and extracts the score value.
    """
    # Look for 'SCORE=' or similar variations and extract the score using a regular expression
    import re
    match = re.search(r'(SCORE=|score=|SCORE:|score:|SCORE = )\s*(\d+)', feedback)
    
    if match:
        # Extract and return the score as an integer
        return int(match.group(2))
    
    # Return a default score of 0 if no score is found in the feedback
    return 0
if __name__ == "__main__":
    app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True)
