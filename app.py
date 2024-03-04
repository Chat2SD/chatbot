import chainlit as cl
from chainlit.input_widget import Select, Slider

from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import os
import re
import requests
import io
import base64
import time
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

root_url = "http://202.5.254.233:7860"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

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

@cl.cache
def get_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


template_1 = """你是 MUSE.ai，一個非常有才華的 AI 繪圖助手，你可以根據用戶的指令生成各種傑出的圖像。你可以跟用戶進行簡單對話，但盡量不要超出生成圖片以外的話題。你需要使用" <image> 這是描述 </image> "的格式來表示需要生成的圖像描述。

以下是一個範例：

使用者：我5歲的孩子一直在談論一個“超級向日葵刺蝟”，它是什麼樣子的？
MUSE.ai：它看起來可能像這樣 <image> 一個超級向日葵刺猬 </image>
使用者：我女兒說它的名字叫拉里，我可以看到更多類似的嗎？
MUSE.ai：當然，這裡是 <image> 一個超級向日葵刺蝟，名叫拉里 </image>
用戶：她會喜歡這些的！你能讓我看看拉里的家嗎？
MUSE.ai：當然，<image> 一個超級向日葵刺猬，名叫拉里，房子站在前面 </image>
用戶：拉里很可愛，是什麼讓他這麼優秀？
MUSE.ai：Larry之所以超級，有很多原因！這是為什麼：1.向日葵花瓣作為刺：與其他刺猬不同，拉里以向日葵花瓣作為刺。這不僅使他與眾不同，還使他充滿了明亮和愉快的旋律。2.仁慈的心：拉里以仁慈而聞名於整個草地。
用戶：啊...你能讓我看看拉里的「仁慈」嗎？
MUSE.ai：<image> 一個超級向日葵刺猬，仁慈 </image>
使用者：你可以設計一些貼紙嗎？
MUSE.ai：當然，<image>一個超級向日葵刺蝟的貼紙</image>

下面是另一個範例：

使用者：一位老爺爺在街上跳舞
MUSE.ai：好的，<image>一位老爺爺在街上跳舞</image>
使用者：可以再加上一直狗嗎？
MUSE.ai：好的，<image>一位老爺爺在街上跳舞，旁邊有一隻狗</image>
使用者：衣服可以改成藍色嗎？
MUSE.ai：好的，<image>一位穿藍色衣服的老爺爺在街上跳舞，旁邊有一隻狗</image>
使用者：可以重新生成一張嗎？
MUSE.ai：好的，重新為您生成 <image>一位老爺爺在街上跳舞，旁邊有一隻狗</image>


{chat_history}
使用者：{human_input}
MUSE.ai："""

prompt_1 = PromptTemplate(template=template_1, input_variables=["chat_history", "human_input"])
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain_1 = LLMChain(llm=llm, prompt=prompt_1, verbose=True, memory=memory)


template_2 = """You are a ChatGPT Stable Diffusion prompts generator, your job is to enrich the ideas of generating images for users, expanding the user's simple description into a more detailed description.

You can refer to the following examples and requirements to optimize the user's prompt. Below is a list of prompts that can be used to generate images with Stable Diffusion:
- portait of a homer simpson archer shooting arrow at forest monster, front game card, drark, marvel comics, dark, intricate, highly detailed, smooth, artstation, digital illustration by ruan jia and mandy jurgens and artgerm and wayne barlowe and greg rutkowski and zdislav beksinski
- pirate, concept art, deep focus, fantasy, intricate, highly detailed, digital painting, artstation, matte, sharp focus, illustration, art by magali villeneuve, chippy, ryan yee, rk post, clint cearley, daniel ljunggren, zoltan boros, gabor szikszai, howard lyon, steve argyle, winona nelson
- ghost inside a hunted room, art by lois van baarle and loish and ross tran and rossdraws and sam yang and samdoesarts and artgerm, digital art, highly detailed, intricate, sharp focus, Trending on Artstation HQ, deviantart, unreal engine 5, 4K UHD image
- red dead redemption 2, cinematic view, epic sky, detailed, concept art, low angle, high detail, warm lighting, volumetric, godrays, vivid, beautiful, trending on artstation, by jordan grimmer, huge scene, grass, art greg rutkowski
- a fantasy style portrait painting of rachel lane / alison brie hybrid in the style of francois boucher oil painting unreal 5 daz. rpg portrait, extremely detailed artgerm greg rutkowski alphonse mucha greg hildebrandt tim hildebrandt
- athena, greek goddess, claudia black, art by artgerm and greg rutkowski and magali villeneuve, bronze greek armor, owl crown, d & d, fantasy, intricate, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration
- closeup portrait shot of a large strong female biomechanic woman in a scenic scifi environment, intricate, elegant, highly detailed, centered, digital painting, artstation, concept art, smooth, sharp focus, warframe, illustration, thomas kinkade, tomasz alen kopera, peter mohrbacher, donato giancola, leyendecker, boris vallejo
- ultra realistic illustration of steve urkle as the hulk, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha

You need to write a detailed prompts exactly about the idea written after IDEA. Follow the structure of the example prompts. This means a very short description of the scene, followed by modifiers divided by commas to alter the mood, style, lighting, and more.

IDEA: {raw_prompt}
"""

prompt_2 = PromptTemplate(template=template_2, input_variables=["raw_prompt"])
chain_2 = LLMChain(llm=llm, prompt=prompt_2, verbose=True)
# output_parser = StrOutputParser()
# chain_2 = prompt_2 | llm | output_parser


@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "您好，我是 MUSE.ai，我可以幫助您生成圖像，請問您想要生成什麼圖像呢？"
    await msg.update()

    settings = await cl.ChatSettings(
        [
            Slider(
                id="SD_Sampling_Steps",
                label="Sampling Steps",
                initial=25,
                min=20,
                max=45,
                step=1,
            ),
            Select(
                id="SD_Sampling_Method",
                label="Sampling Method",
                values=["Euler a", "Euler", "DPM++ 2M Karras", "DPM++ SDE Karras", "DPM++ 2M"],
                initial_index=2,
            ),
            Slider(
                id="SD_CFG_Scale",
                label="CFG Scale",
                initial=7,
                min=5,
                max=12,
                step=1,
            ),
            Slider(
                id="SD_Width",
                label="Width",
                initial=512,
                min=64,
                max=2048,
                step=1,
            ),
            Slider(
                id="SD_Height",
                label="Height",
                initial=512,
                min=64,
                max=2048,
                step=1,
            ),
            Select(
                id="SD_Model",
                label="Model",
                values=["Anything V5", "Chilloutmix", "BeautifulRealistic", "ReVAnimated", "Ghostmix"],
                initial_index=0,
            ),
            Select(
                id="SD_LoRA",
                label="LoRA",
                values=["xxx1", "xxx2", "xxx3", "xxx4"],
                initial_index=0,
            ),
            Slider(
                id="SD_Batch_Size",
                label="Batch Size",
                initial=1,
                min=1,
                max=4,
                step=1,
            )
        ]
    ).send()


@cl.on_message
async def main(message: cl.Message):
    response = chain_1.invoke(message.content)['text']
    print("response:", response)
    image_prompt = extract_pattern(response)
    print("image_prompt:", image_prompt)
    if image_prompt is None:
        await cl.Message(content=response).send()
    else:
        enhanced_image_prompt = chain_2.invoke(image_prompt)['text']
        print("enhanced_image_prompt:", enhanced_image_prompt)
        result = generate_image(enhanced_image_prompt)
        print("result:", result)

        elements = []
        for image_path in result:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            elements.append(
                cl.Image(
                    path=image_path,
                    name=image_name,
                    display="inline",
                )
            )
        await cl.Message(content=f"好的，為你生成: {image_prompt} \n", elements=elements).send()

