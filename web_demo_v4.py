import torch
import requests
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, request, Response
# from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
import gradio as gr
import os, sys, io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.constants import LOGDIR

import hashlib
from transformers import TextStreamer, TextIteratorStreamer
from threading import Thread

from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import base64
import time
import re
import cv2
import matplotlib.font_manager

import spacy
nlp_zh = spacy.load("zh_core_web_sm")
nlp = spacy.load("en_core_web_sm")

import logging
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--english", action='store_true', help='only output English')
parser.add_argument("--version", type=str, default="chat", help='version to interact with')
parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
parser.add_argument("--no_prompt", action='store_true', help='Sometimes there is no prompt in stage 1')
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
parser.add_argument("--server_port", type=int, default=80, help='the gradio server port')
args = parser.parse_args()


# model_path = "/root/jinyfeng/models/LLaVa/llava-v1.5-13b"
model_path = '/root/jinyfeng/models/LLaVa/llava-v1.6-34b'

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "load_8bit": args.load_8bit,
    "load_4bit": args.load_4bit,
    "conv_mode": None,
    "sep": ",",
    # "temperature": 0,
    "temperature": 0.2,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "server_port": args.server_port
})()

from llava.utils import (build_logger, server_error_msg, violates_moderation, moderation_msg)
from llava.utils import disable_torch_init
disable_torch_init()
from enum import auto, Enum
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()

class ERROR_CODE(Enum):
    SUCCESS = 0
    TIME_OUT = 10001
    IMAGE_ERROR = 10002
    PARAM_ERROR = 10003
    MODEL_INF_ERROR = 10004

model_name = get_model_name_from_path(args.model_path)
device_count=torch.cuda.device_count() 
print('device_count========',device_count)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# device_fp16=torch.device("cuda")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=args.model_path,
    model_base=args.model_base,
    model_name=model_name,
    load_8bit=args.load_8bit,
    # load_4bit=args.load_4bit,
    # load_8bit=True, 
    load_4bit=True,
)

if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "mistral" in model_name.lower():
    conv_mode = "mistral_instruct"
elif "v1.6-34b" in model_name.lower():
    conv_mode = "chatml_direct"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"
if args.conv_mode is not None and conv_mode != args.conv_mode:
    print(
        "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
            conv_mode, args.conv_mode, args.conv_mode
        )
    )
else:
    args.conv_mode = conv_mode


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def load_url_image(image_url):
    try:
        # imgData = requests.get(file_prompt)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36'}
        time_start = time.time()
        # imgData = requests.get(image_url, headers=headers, stream=True)
        imgData = requests.get(image_url)
        process_time = '{0:.2f}'.format((time.time() - time_start) * 1000)
        print('process_time, imgData.status_code========', process_time, imgData.status_code)
        # image = Image.open(io.BytesIO(image)).convert("RGB")
        images = [Image.open(io.BytesIO(imgData.content)).convert("RGB")]
        return images
    except Exception as e:
        print("error message111111", e)
        time.sleep(3)
        try:
            time_start = time.time()
            imgData = requests.get(image_url)
            process_time = '{0:.2f}'.format((time.time() - time_start) * 1000)
            print('process_time, imgData.status_code==========', process_time, imgData.status_code)
            images = [Image.open(io.BytesIO(imgData.content)).convert("RGB")]
            return images
        except Exception as e:
            print("error message222222", e)
            images=[]
            return images

def large_model_post(
        file_prompt,
        input_text,
        temperature=0.2,
        top_p=0.7,
        num_beams=1,
        token_len=1024,
        streamer_out=True
        ):

    # inp = input_text
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + input_text
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + input_text
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], inp)
    # image = None
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print('prompt=================', prompt)

    streamer=None
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    if streamer_out:
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30)
    
    start_str = '<|startoftext|>'
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # print('stop_str===========', stop_str)

    # result_text = []
    ret_code=0
    images = load_url_image(file_prompt)
    if len(images)<1:
        ret_code=ERROR_CODE.IMAGE_ERROR
        yield json.dumps({"code": ret_code.value, "result": ret_code.name, "message": ret_code.name})
    else:
        try:
            # image_size = image.size
            image_sizes = [image.size for image in images]
            # Similar operation in model_worker.py

            image_tensor = process_images(images, image_processor, model.config)
            images = image_tensor.to(model.device, dtype=torch.float16)
            image_args = {"images": images, "image_sizes": image_sizes}

            # if type(image_tensor) is list:
            #     image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            # else:
            #     image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            
            with torch.inference_mode():
                if not streamer_out:
                    print('streamer_out is False')
                    output_ids = model.generate(
                        input_ids,
                        images=images,
                        image_sizes=image_sizes,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        # num_beams=num_beams,
                        # repetition_penalty=1.3,
                        no_repeat_ngram_size=3,
                        max_new_tokens=token_len,
                        use_cache=True)
                    outputs = tokenizer.decode(output_ids[0]).strip()
                    # outputs = tokenizer.decode(output_ids[0])
                    # outputs = outputs.strip()
                    if outputs.startswith(start_str):
                        outputs = outputs[len(start_str):]
                    if outputs.endswith(stop_str):
                        outputs = outputs[: -len(stop_str)]
                    print('outputs=========', outputs)
                    # result_text.append((input_text, outputs))
                    # return ret_code, outputs
                    yield json.dumps({"code": ret_code, "result": outputs, "message": "SUCCESS"})

                else:
                    thread = Thread(target=model.generate, kwargs=dict(
                        inputs=input_ids,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        # num_beams=num_beams,
                        # repetition_penalty=1.3,
                        no_repeat_ngram_size=3,
                        max_new_tokens=token_len,
                        streamer=streamer,
                        use_cache=True,
                        **image_args
                    ))
                    thread.start()
                    generated_text = ""
                    print('stop_str===========', stop_str)
                    for new_text in streamer:
                        # print('new_text===========', new_text)
                        if new_text.endswith(stop_str):
                            new_text = new_text[:-len(stop_str)].strip()
                        generated_text += new_text
                        yield json.dumps({"code": ret_code, "result": new_text, "message": "SUCCESS"})
                        time.sleep(0.001)                    
                        # # yield json.dumps({"code": ret_code, "result": generated_text, "message": "SUCCESS"}).encode() + b"\0"
                        # yield json.dumps({"result": generated_text, "ret_code": ret_code})
                    print('generated_text==========', generated_text)
                    thread.join()
                
        except Exception as e:
            print("error message", e)
            # result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
            ret_code=ERROR_CODE.MODEL_INF_ERROR
            yield json.dumps({"code": ret_code.value, "result": ret_code.name, "message": ret_code.name})
            # return ret_code, "result_text"


# @app.route('/api/large_model_post', methods=['POST'])
@app.route('/api/large_model_stream', methods=['POST'])
def api_large_model_post():
    request_data = request.get_json()
    print('request_data===========', request_data)

    if 'file_prompt' not in request_data or 'input_text' not in request_data:
        ret_code=ERROR_CODE.PARAM_ERROR
        return json.dumps({"code": ret_code.value, "result": ret_code.name, "message": ret_code.name})

    file_prompt = request_data['file_prompt']
    # print('file_prompt===========', file_prompt)
    input_text = request_data['input_text']
    # print('input_text===========', input_text)
    temperature, top_p, token_len, streamer_out=0.2, 0.7, 1024 , True
    if 'streamer_out' in request_data:
        streamer_out_in = request_data['streamer_out']
        streamer_out = streamer_out_in == str(True)
        print('streamer_out==========', streamer_out)
    if 'temperature' in request_data:
        temperature = float(request_data['temperature'])
        print('temperature==========', temperature)
    if 'top_p' in request_data:
        top_p = float(request_data['top_p'])
        print('top_p==========', top_p)
    if 'token_len' in request_data:
        token_len = min(int(request_data['token_len'], 256), 1024)
        print('token_len==========', token_len)
    
    if streamer_out:
        print('streamer_out is True')
        generator = large_model_post(file_prompt=file_prompt, 
                                    input_text=input_text, 
                                    temperature=temperature,
                                    top_p=top_p,
                                    token_len=token_len,
                                    streamer_out=streamer_out)
        return Response(generator, mimetype='text/plain')
        # return generator, {"Content-Type": "text/plain"}
        # return app.response_class(generator, mimetype='text/plain')
    else:
        response = large_model_post(file_prompt=file_prompt, 
                                    input_text=input_text, 
                                    temperature=temperature,
                                    top_p=top_p,
                                    token_len=token_len,
                                    streamer_out=streamer_out)
        return response
    # response = large_model_post(file_prompt, input_text)
    # print('response=======', response.__next__())
    # # ret_datas = {
    # #     "ret_code": ret_code,
    # #     "result": result
    # # }    
    # # return jsonify(ret_datas)
    # # return json.dumps(ret_datas)


default_chatbox = [("", "Hi, What do you want to know about from llava?")]
def clear_fn(value):
    return "", default_chatbox, None, None

def clear_fn2(value):
    return default_chatbox, None


if __name__ == '__main__':
    gr.close_all()
    examples = []
    ret_code=0
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('<h1><center>LLava-1.6-34b Model</center></h1>')
        with gr.Row():
            temperature = gr.Slider(maximum=1, value=0.2, minimum=0, label='Temperature')
            top_p = gr.Slider(maximum=1, value=0.7, minimum=0, label='Top P')
            token_len = gr.Slider(maximum=1024, value=1024, minimum=0, step=64, label='Max output tokens')

        input_text_image = gr.Textbox(label='Enter the question you want to know',
                                value='分析图片中的内容，识别并描述图片的细节，包括人物特征（如数量、外观、年龄、服装、服装颜色、服装品牌、表情和行为、是否戴口罩、是否背包、是否抽烟、是否在玩手机）、车辆特征（车辆外形、车辆颜色、车辆品牌）、场景布局（比如地点、物体和活动）、以及时间和可能的情境背景。请为我提供一份简要的报告，总结图片中的关键元素。',
                                elem_id='textbox')
        with gr.Row():
            with gr.Column(scale=4):
                # input_image_file = gr.Image(type='pil', label='Input Image')
                input_image_file = gr.Image(type='filepath', label='Input Image')      
            with gr.Column(scale=6):
                result_text_image = gr.components.Chatbot(label='Conversation History', 
                                            # label='Multi-round conversation History', 
                                            value=[("", "Hi, What do you want to know about?")], 
                                            height=500)
                result_code = gr.Textbox(visible=False)
        # hidden_image_hash = gr.Textbox(visible=False)
        with gr.Row():
            image_submit = gr.Button('Submit')
            image_clear = gr.Button('Clear')

        print(gr.__version__)
        image_submit.click(fn=large_model_post,
                                inputs=[input_image_file, input_text_image, temperature, top_p, token_len],
                                outputs=[result_code, result_text_image])
        image_clear.click(lambda: [[], '', ''], None,
                                [input_image_file, input_text_image, result_text_image])

    # gr.Interface(fn=large_model_post, inputs="text", outputs="text").launch(server_name="0.0.0.0", show_error=True, server_port=args.server_port)
    # app.run(debug=True)
    # app.logger.setLevel(logging.INFO)
    app.run(host="0.0.0.0", port=args.server_port)
    # app.run(host="0.0.0.0", port=args.server_port, debug=True)

    # demo.queue(concurrency_count=5)
    # # demo.launch(share=True)
    # demo.launch(server_name="0.0.0.0", show_error=True, server_port=args.server_port)


