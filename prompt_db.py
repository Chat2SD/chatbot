import os
import random
import numpy as np
import pandas as pd

import unicodedata
import time

#need to 'pip install sentence-transformers" by youself
import sys
sys.path.append('C:\\Kaggle\\StableDiffusion\\sentence-transformers\\sentence-transformers')
from sentence_transformers import SentenceTransformer ,util

import warnings
warnings.filterwarnings('ignore')
  
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

seed_everything(42)   

def is_english_only(string):
    for s in string:
        cat = unicodedata.category(s)         
        if not cat in ['Ll', 'Lu', 'Nd', 'Po', 'Pd', 'Zs']:
            return False
    return True

           


#PROMPT_CSV_PATH = "train_clusters_2lfiter_th0p8_340k.csv"
PROMPT_CSV_V2_PATH = "diffusionDB2M_with_cfg_filter_600k.csv"
PERSIST_STORE_DB = "chroma_db_diffusionDB_prompt600k.db"
#PERSIST_STORE_DB_V2 = "chroma_db_diffusionDB2M_with_cfg"


#langchain & chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)    
from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter



class ImagePromptDB:
    def __init__(self):
        self.vectordb = None
        #self.promptdb = pd.read_csv('train_clusters_2lfiter_th0p8_340k.csv') #60k
        self.promptdb = pd.read_csv(PROMPT_CSV_V2_PATH) #60k
    
    def load_vectordb(self,dbname="chroma"):
        # create the open-source embedding function
        if(dbname == "chroma"):
            embedding_function = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

            self.vectordb = Chroma(persist_directory=PERSIST_STORE_DB, embedding_function=embedding_function,collection_metadata={"hnsw:space":"cosine"})
            new_prompts ,indexes,scores= promptdb.get_similar_prompts("db should work efficiently",top_k=1)


    def load_rawdb(self,dbname):
        print('load_rawdb: To be implemented')

    def get_similar_prompts (self,query, top_k=4):
        docs = self.vectordb.similarity_search_with_score(query,k=top_k)
        list_prompts = []
        list_indexes = []
        list_scores = []
        for i in range(len(docs)):
            list_prompts.append(docs[i][0].page_content.replace("prompt: ",""))  
            list_indexes.append(docs[i][0].metadata['row']) 
            list_scores.append(docs[i][1])#for score  
        return list_prompts ,list_indexes ,list_scores         

    def get_info_by_index (self,index):
        return self.promptdb.iloc[index]

    # def get_similar_prompts_with_cfg (self,query, top_k=4):
    #     print('get_similar_prompts_with_cfg: To be implemented' )
    #     docs = self.vectordb.similarity_search_with_score(query,k=top_k)


def make_csv_db_with_cfg(dbpath = PROMPT_CSV_V2_PATH ):

    #https://huggingface.co/datasets/poloclub/diffusiondb/tree/main
    df = pd.read_parquet('DiffusionDB2M/metadata.parquet')

    #---preprocessing-----------
    #print(len(df)) #2000000 2M
    #df = df[(df['width'] == 512) & (df['height'] == 512)]
    df['prompt'] = df['prompt'].str.strip()
    df = df[df['prompt'].notnull()] #maybe unicode issue 
    df = df[df['prompt'].map(lambda x: len(x.split())) >= 5]
    df = df[~df['prompt'].str.contains('^(?:\s*|NULL|null|NaN)$', na=True)]
    #df = df[df['prompt'].apply(is_english_only)]
    df['head'] = df['prompt'].str[:40]
    df['tail'] = df['prompt'].str[-40:]
    df.drop_duplicates(subset='head', inplace=True)
    df.drop_duplicates(subset='tail', inplace=True)
    df.reset_index(drop=True, inplace=True)

    #----------------------------


    #create image link here
    # for i in tqdm(range(1, 2000, 100)):
    #     image_dir = f'/kaggle/input/diffusiondb-2m-part-{str(i).zfill(4)}-to-{str(i+99).zfill(4)}-of-2000/'
    #     images = os.listdir(image_dir)
    #     df.loc[df['image_name'].isin(images), 'filepath'] = image_dir + df['image_name']
    print("finally ")
    print(len(df))
    #full fitering : 154320
    print (df.head()) #606805 ?
    #df[:10].to_csv("prompt_with_cfg_10.csv",index=False,encoding='utf-8')  #table csv need header 
    df.to_csv(dbpath,index=False,encoding='utf-8')  #table csv need header 

    #--------------------------------------------------------------------

def make_persist_db(dbpath = PERSIST_STORE_DB):
    df = pd.read_csv(PROMPT_CSV_V2_PATH) #60k
    print(len(df)) #606804

    #----use unique flag to reduce data to 340k
    #df =df[df.unique==1]
    #print("unique count is " )
    #print(len(df)) 344341

    #preprocessing
    #only mix dataset(+2.47m) required to preprocess again
    #diffusionDB (train_clusters_2lfiter_th0p8_340k.csv) done
    # #df = df[(df['width'] == 512) & (df['height'] == 512)]
    # print(len(df)) #1043407
    # #print(df['prompt'])
    # df['prompt'] = df['prompt'].str.strip()
    # #print(len(df['prompt']))
    # print(len(df))
    # df = df[df['prompt'].notnull()] #maybe unicode issue 
    # print(len(df)) #1999517
    # df = df[df['prompt'].map(lambda x: len(x.split())) >= 5]

    # #df = df[~df['prompt'].str.contains('^(?:\s*|NULL|null|NaN)$', na=True)]
    # print(len(df))#1891338
    # #df = df[df['prompt'].apply(is_english_only)]

    # #df.drop_duplicates(subset='head', inplace=True)
    # print("total len ")
    # df.reset_index(drop=True, inplace=True)
    # print(len(df)) #15  

    # #---------------------save csv----------------------------------
    embedding_function = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    df['prompt'].to_csv("prompt_text_600k.csv",index=False,encoding='utf-8')  #table csv need header 
    print(df['prompt'][:20])
    #----------------------------------------------------------
    #-------------save db-----------------------------------------------------------
    loader = CSVLoader("prompt_text_600k.csv", encoding='utf8')
    documents = loader.load()
    docs =documents  

    # load it into Chroma
    db = Chroma.from_documents(docs, embedding_function, persist_directory=dbpath,collection_metadata={"hnsw:space":"cosine"},)
    #--------------------------------------------------------------------


if __name__ == '__main__':

    # #-------make db---------
    # make_csv_db_with_cfg()
    # make_persist_db()
    # #--------------------

    #--------init db------------
    promptdb = ImagePromptDB()
    promptdb.load_vectordb()



    #-----query 1--------
    print("-------query 1 ------- " )
    #query = "一隻狗在廚房"
    query = "a dog in the kitchen"

    start_time = time.time()

    print("query is " +query  )

    new_prompts,indexes,scores = promptdb.get_similar_prompts(query)

    print(new_prompts )
    print(new_prompts[0])
    print(indexes)
    print(scores)
    end_time = time.time()
    elapsed_time = end_time - start_time    
    print("query 1 耗时:", elapsed_time, "秒")

    #get info by key(index)
    print(promptdb.get_info_by_index(indexes[0]))

    #-----query 1--------


    #-----query 2--------
    print("-------query 2 ------- " )
    start_time = time.time()

    #query = "一隻狗在舉重"
    #query = "一隻狗在進行舉重運動"
    #query = "一隻狗在舉重,精確,最高品質"
    #query = "一隻狗在舉重,精確,油畫"
    #query = "一隻狗在舉重,精確"
    #query = "a dog doing weights"
    #query = "a dog doing weights,detail"
    #query ="a dog lifting weights"
    #query = "a dog doing weights lifting"
    query = "一隻狗有六隻腳"  #'a dog with eight legs' # it is not ok for user :(
    print("query is " +query  )

    new_prompts ,indexes,scores= promptdb.get_similar_prompts(query,top_k=10)

    print(new_prompts )
    print(new_prompts[0])
    print(indexes)
    print(scores)
    end_time = time.time()
    elapsed_time = end_time - start_time    
    print("query 2 耗时:", elapsed_time, "秒")
    #get info by key(index)
    print(promptdb.get_info_by_index(indexes[0]))

