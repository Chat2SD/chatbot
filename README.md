# chatbot

## 環境安裝
```shell
git clone https://github.com/Chat2SD/chatbot
cd chatbot
conda create -n chat2sd python=3.11
pip install -r requirements.txt
```

## 環境變數
新建一個 **.env**，設定好 OPENAI_API_KEY = YOUR_OPENAI_API_KEY


## 啟動 app
```shell
chainlit run app.py -w -h
```
