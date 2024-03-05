from chainlit.input_widget import Select, Slider
import chainlit as cl
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import *
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


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
                initial_index=1,
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

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    with open("prompts/prompt-v1-cn.txt") as f:
        template_1 = f.read()

    prompt_1 = PromptTemplate(template=template_1, input_variables=["chat_history", "human_input"])
    memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)
    chain_1 = LLMChain(llm=llm, prompt=prompt_1, verbose=True, memory=memory)

    with open("prompts/prompt-v2-cn.txt") as f:
        template_2 = f.read()

    prompt_2 = PromptTemplate(template=template_2, input_variables=["raw_prompt"])
    chain_2 = LLMChain(llm=llm, prompt=prompt_2, verbose=True)
    # output_parser = StrOutputParser()
    # chain_2 = prompt_2 | llm | output_parser

    cl.user_session.set("chain_1", chain_1)
    cl.user_session.set("chain_2", chain_2)


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    chain_1 = cl.user_session.get("chain_1")
    chain_2 = cl.user_session.get("chain_2")

    response = await cl.make_async(chain_1.invoke)(
        input=message.content, callbacks=[cl.LangchainCallbackHandler()]
    )
    print("response:", response["text"])
    image_prompt = extract_pattern(response["text"])
    print("image_prompt:", image_prompt)

    if image_prompt is None:
        msg.content = response["text"]
        await msg.update()
    else:
        msg.content = f"好的，正在為你生成：{image_prompt}。請稍候..."
        await msg.update()

        enhanced_image_prompt = await cl.make_async(chain_2.invoke)(
            input=image_prompt, callbacks=[cl.LangchainCallbackHandler()]
        )
        print("enhanced_image_prompt:", enhanced_image_prompt["text"])
        result = generate_image(enhanced_image_prompt["text"])
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

        msg.elements = elements
        await msg.update()