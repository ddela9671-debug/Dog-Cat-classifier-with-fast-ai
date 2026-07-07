import gradio as gr
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learn = load_learner("model.pkl")

categories=('Dog','Cat')

def classify_image(img):
    pred, pred_idx, probs = learn.predict(img)
    return {
        learn.dls.vocab[i]: float(probs[i])
        for i in range(len(probs))
    }

image = gr.Image(type="pil")
label = gr.Label()
examples = ['dog.jpg', 'gato.jpg', 'gorilla.webp','leon.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
