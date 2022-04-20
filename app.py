from flask import Flask, render_template, request
from signdetect import detect
from PIL import Image
import cv2
app = Flask(__name__)
import os
import openai

def auto_sentence(words):
	# DONT RUN THIS MULTIPLE TIMES, FEW FREE CHANCES ONLY
	openai.api_key = "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"
	p = "CorrecCt this to standard English:\n\n"+words+"."
	response = openai.Completion.create(engine="text-davinci-002",
										prompt=p, temperature=0,
										max_tokens=60, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0
										)
	return (response.choices[0].text[2:])

def predict_label(video_path):

	prediction = detect(video_path)
	l = []
	for i in prediction.keys():
		if (prediction[i] > 20):
			l.append(i)
	prediction = " ".join(l)

	#dont run unless necessary
	#prediction = auto_sentence(prediction)

	return prediction

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		video = request.files['my_video']
		video_path = "static/" + video.filename
		video.save(video_path)
		p = predict_label(video_path)
	return render_template("index.html", prediction=p, video_path=video_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)