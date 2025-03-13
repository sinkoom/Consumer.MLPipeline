import os
from azure.identity import ClientSecretCredential
import yaml
import openai
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import sqlite3
import pandas as pd
from langchain.agents import create_sql_agent, AgentExecutor, Tool, ZeroShotAgent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain
from langchain_experimental.tools import PythonREPLTool
import pickle
import pyreadr
import json
config_file = "/Users/L052033/Downloads/config.yaml"
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

tenant_id = config['environment']['TENANT_ID']
print(tenant_id)

def get_credentials():
    tenant_id = config['environment']['TENANT_ID'] 
    client_id = config['environment']["CLIENT_ID"]
    client_secret = config['environment']["CLIENT_SECRET"]
    credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    return credential

credential = get_credentials()
token = credential.get_token("https://cognitiveservices.azure.com/.default")

os.environ["OPENAI_API_TYPE"] = "azure_ad"
os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://cwb-openai-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = token.token
api_key = os.environ["AZURE_OPENAI_API_KEY"]

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

connection = sqlite3.connect("mmmVolume.db")

#dashboard_file = "/Users/L052033/Downloads/mounjaro_weekly_hcp_data_20241002.rds" # rds file requires pyread
dashboard_file = "/Users/L052033/Downloads/taltz_weekly_dma_google_20241029 1.csv"

# Replace 'your_file.rds' with the path to your .rds file
#result = pyreadr.read_r(dashboard_file)
#df = result[None]  # None is used to get the default object in the file

ml_model_obj = "/Users/L052033/Downloads/model_tst.pkl"

policy_file = "/Users/L052033/Downloads/mmm_policy.json"
df = pd.read_csv(dashboard_file)
# Load the policy file
with open(policy_file, 'r') as file:
    policy = json.load(file)
df.to_sql("mmmVolumeTable", connection, if_exists='replace')
print(df.head())

db = SQLDatabase.from_uri('sqlite:///mmmVolume.db')

def custom_parse_output(output):
    try:
        parsed_output = some_parsing_function(output)
        return parsed_output
    except Exception as e:
        print(f"Parsing error: {e}")
        return "Sorry, I couldn't understand the response. Please try again."

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    allow_dangerous_requests=True,
    agent_executor_kwargs=dict(handle_parsing_errors=True),
    verbose=True,
    return_intermediate_steps=True,
    top_k=10
)

template = PromptTemplate(
    input_variables=["user_inquiry", "background_info"],
    template="""{background_info}
    Question: {user_inquiry}
    """
)

description = (
    "Useful when you require to answer analytical questions about the marketing mix models (MMM). "
    "Use this with the Python REPL tool if the question is about customer analytics, market trend data like source of authority, new to brand (nbrx) volumes and scripts"
    "like 'What is the total number of new to brand (nbrx) scripts for Mounjaro?' or 'count the number of scripts by dma (dma_cd) group'. "
    "or what are the market trends for brand volume or brand interest data"
    "use the pickle model object for forecasting if you need any forecast predictions"
    "Try not to use clause and limit in the SQL."
)

mmmVolume_data_tool = Tool(
    name="MMMVolume",
    func=agent_executor.run,
    description=description,
)

tools = [PythonREPLTool()]
tools.append(mmmVolume_data_tool)

prefix = "Below are tools that you can access:"
suffix = (
    "Pass the relevant part of the request directly to the MMM tool.\n\n"
    "Request: {input}\n"
    "{agent_scratchpad}"
)

agent_prompt = ZeroShotAgent.create_prompt(
    tools, prefix=prefix, suffix=suffix, input_variables=["input", "agent_scratchpad"]
)

zero_shot_agent = ZeroShotAgent(
    llm_chain=LLMChain(llm=llm, prompt=agent_prompt),
    allowed_tools=[tool.name for tool in tools]
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=zero_shot_agent, tools=tools, verbose=True, handle_parsing_errors=True
)

with open(ml_model_obj, 'rb') as file:
    model = pickle.load(file)

def query_model(input_data):
    prediction = model.predict([input_data])
    return prediction[0]

model_tool = Tool(name="query_model", func=query_model,
description="This tool queries the machine learning model with input data and returns the prediction.")

background_info = """
As the customer analyst, my role is to analyze the volume and impression patterns of advertisement channels. The feature engineering in table 'mmmVolumeTable' is crucial for statistical exploration. For example:
- column 'date' can be grouped into bins of weeks and months ranges, such as - Week 2023-01: 2127 - Week 2023-02: 2516 and so on. The column 'dma_cd' is the designated market code showing different geographic locations of the united states. 
They can be grouped as 500, 501 and so on. Other columns are described as follows:
nbrx_brand: volumetric data indicating the number of scripts written for the taltz drug and brand (an atopic dermatitis drug), 
nbrx_mkt: the volume of scripts of similar drugs from different pharmaceutical industries, 
google_market: the total number of google searches for the atopic dermatitis drugs which include taltz and 
google_taltz: the total number of google searches for taltz brand  ,  
Understanding the data in these columns helps us gain insights about our advertisement and the taltz brand, enabling us to offer personalized services and develop effective marketing strategies for marketing the drug.
Use all data at all times
For Taltz Q2 2023 data, ott had a high importance (influence) in driving nbrx


"""

#user_inquiry = "Tell me about the data trends for nbrx_mkt and nbrx_brand. Aggregate data at the monthly level"

user_inquiry = "Filter dma_cd for 501 and use the monthly aggregated data for the analysis. Imagine nbrx_mkt, nbrx_brand, google_market and google_taltz are related. Using a structural causal model and the data to tell if there is a causal effect and relationship between variables. Use regression models from the python agent for the analysis. plot the graph structure for the data "

def answer_question(user_inquiry, background_info):
    # Check for special scenarios
    if "urgent" in user_inquiry:
        response = "This seems urgent. Let me prioritize addressing your concern."
    elif all(word in user_inquiry for word in ["brand", "market basket"]):
        response = "I can provide information on market baskets, but I'll exclude brand details."
    
    elif all(word in user_inquiry for word in ["brand", "market basket", "drug", "disease"]):
        ordered_list = ["brand", "market basket", "drug", "disease"]
        response = f"The keywords have been ordered as follows: {ordered_list}. Now, let me address your question."

    else:
        # Pre-process the question based on the policy
        if any(topic in user_inquiry for topic in policy['prohibited_topics']):
            return policy['error_handling']['default_response']

        # Generate the response
        response = agent_executor.run(template.format(background_info=background_info, user_inquiry=user_inquiry))

        # Post-process the response based on the policy
        #if len(response) > policy['response_structure']['max_length']:
        #    response = response[:policy['response_structure']['max_length']] + "..."

    return response

print(answer_question(user_inquiry, background_info))