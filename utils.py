import re
import requests
import io
import base64
import time
from PIL import Image
from datetime import datetime
import os
import chainlit as cl

root_url = "http://202.5.254.233:7860"

def extract_pattern(message, pattern=r"<image>(.*?)<\/image>"):
    matches = re.findall(pattern, message, re.DOTALL)
    for match in matches:
        return match.strip()
    return None

def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

def generate_image(prompt: str):
    cl_chat_settings = cl.user_session.get("chat_settings")
    print(cl_chat_settings)

    url = f'{root_url}/sdapi/v1/txt2img'
    payload = {
        "prompt": prompt,
        "negative_prompt": "EasyNegative",
        "steps": int(cl_chat_settings["SD_Sampling_Steps"]),
        "sampler_name": cl_chat_settings["SD_Sampling_Method"],
        "width": int(cl_chat_settings["SD_Width"]),
        "height": int(cl_chat_settings["SD_Height"]),
        "cfg_scale": cl_chat_settings["SD_CFG_Scale"],
        "batch_size": int(cl_chat_settings["SD_Batch_Size"]),
    }
    try:
        response = requests.post(url=url, json=payload)
        r = response.json()
    except:
        raise requests.HTTPError("Can't connect to the SD server.")
    else:
        if r.get("images"):
            result = []
            timestamp = get_timestamp()
            dir = f"./output/{timestamp.split('-')[0]}"
            os.makedirs(dir, exist_ok=True)
            for i, img in enumerate(r['images']):
                image = Image.open(io.BytesIO(base64.b64decode(img)))
                image_name = f"txt2img-{timestamp}-{i+1}.jpeg"
                image_path = os.path.join(dir, image_name)
                image.save(image_path)
                result.append(image_path)
            return result
        else:
            raise ValueError("Your request did not generate an image. Please modify the prompt and try again.")