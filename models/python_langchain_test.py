#df
#from langchain_openai import AzureChatOpenAI
import os
import builtins
import dill
from azure.identity import ClientSecretCredential
import yaml
import openai
# Import the ChatOpenAI class
from langchain_openai import ChatOpenAI
import sqlite3
import pandas as pd
# Import the AzureChatOpenAI class
from langchain_openai import AzureChatOpenAI

# Import necessary libraries and modules
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain
from langchain.agents import (AgentExecutor, Tool, ZeroShotAgent)
from langchain_experimental.tools import PythonREPLTool
#from langchain import LangChain
import pickle
from langchain.agents import load_tools
from langchain.tools import Tool
import pyreadr
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from doubt import Boot
config_file = "/Users/L052033/Downloads/config.yaml"
# Load configuration from a YAML file
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# Access the TENANT_ID
tenant_id = config['environment']['TENANT_ID']
print(tenant_id)





print(config)
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


# Initialize the Langchain ChatOpenAI model
#pip install openai langchain-openai langchain-community db-sqlite3 langchain-openai
#llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo-1106")
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o", #gpt-4o "o1_model" "o1_mini"
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=1
)

#read HCP data




# Connect to the SQLite database
connection = sqlite3.connect("mmmVolume.db")

#dashboard_file = "/Users/L052033/Downloads/mounjaro_weekly_hcp_data_20241002.rds" # rds file requires pyread
#dashboard_file = "/Users/L052033/Downloads/taltz_weekly_dma_google_20241029 1.csv"
#ml_model_data = '/Users/L052033/Downloads/OneDrive_2_1-16-2025/Mounjaro_Dashboard_Tables/mounjaro_brand_final.csv'
#ml_model_data = "/Users/L052033/Downloads/MounjaroModelData.csv"
#ml_model_data = "/Users/L052033/Downloads/MounjaroModelDatawDate.csv"
ml_model_data = "/Users/L052033/Downloads/bnn_upd.csv"
dashboard_file = "/Users/L052033/Downloads/roi_main_report.csv"

# Replace 'your_file.rds' with the path to your .rds file
#result = pyreadr.read_r(dashboard_file)
#df = result[None]  # None is used to get the default object in the file

#ml_model_obj = "/Users/L052033/Downloads/model_tst.pkl"


model_files = {
    "ml_model_obj_lower": "/Users/L052033/Downloads/model.dill", #model.dill
    "ml_model_obj_mid": "/Users/L052033/Downloads/model.dill",
    "ml_model_obj_upper": "/Users/L052033/Downloads/model.dill"
}

df_dshbd = pd.read_csv(dashboard_file)
df_mldata = pd.read_csv(ml_model_data)
print(df_mldata.iloc[0])


df_mldata = df_mldata.drop(df_mldata.columns[0], axis=1)
#df = df[df.dma_cd == 501]
print(df_dshbd.head(1))
print(df_mldata.head(1))
print(list(df_mldata.columns))
# Convert DataFrame to a SQLite table named "RetailSalesTable"
df_dshbd.to_sql("mmmVolumeTable", connection, if_exists='replace')
df_mldata.to_sql("mmmModelTable", connection, if_exists='replace')





# Create an instance of SQLDatabase using the 'customer.db' SQLite database
db = SQLDatabase.from_uri('sqlite:///mmmVolume.db')





# Custom function to handle parsing errors
def custom_parse_output(output):
    try:
        # Attempt to parse the output
        parsed_output = some_parsing_function(output)
        return parsed_output
    except Exception as e:
        # Handle parsing error
        print(f"Parsing error: {e}")
        return "Sorry, I couldn't understand the response. Please try again."

# Create an SQL agent executor with specified parameters
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,  # Handle parsing errors
    allow_dangerous_requests=True,
    agent_executor_kwargs=dict(handle_parsing_errors=True),
    verbose=True,
    return_intermediate_steps=True,
    top_k = 1
)

# Define user input
##user_inquiry = "Tell me about how nbrx_mkt is trending. Aggregate data by weeks and months"
#Tell me about data. Generate summary statistics and use causal framework to 
# Run the agent to generate a response
#evaluate the impact of nbrx_brand on nbrx_mkt. Use python tool call as executor and show linear coefficients. Show step by step your causal chain of thought.
#agent_executor.run(user_inquiry)



# Create the prompt template
template = PromptTemplate(
    input_variables=["user_inquiry", "background_info"],
    template="""{background_info}
    Question: {user_inquiry}
    """
)


# Define a description to suggest how to determine the choice of tool
description = (
    "Useful when you require to answer analytical questions about the marketing mix models (MMM). "
    #"Use this more than the Python REPL tool if the question is about customer analytics, market trend data like source of authority, new to brand (nbrx) volumes and scripts"
    #"like 'What is the total number of new to brand (nbrx) scripts for Mounjaro?' or 'count the number of scripts by dma (dma_cd) group'. "
    #"or what are the market trends for brand volume or brand interest data"
    #"use the pickle model object for forecasting if you need any forecast predictions"
    "Incase you have to summarize data use all columns. Eg. brand,run_period, nbrx_market, nbrx_brand, channel engagements, impactable_share, roi, rpe, mroi, impactable_share_diff, impacted_nbrx and revenue_per_nbrx."
    " You can also use mrpe, engagements_previous, engagements_diff, impactable_share_previous, revenue_per_nbrx_previous        revenue_per_nbrx_diff   impacted_revenue        cost_per_engagement  cost_per_engagement_previous    cost_per_engagement_diff        total_cost      roi_previous roi_diff        rpe_previous    rpe_diff        mroi_previous   mroi_diff       mrpe_previous        mrpe_diff"
    "Make sure to remove unnecessary formatting in the query, to avoid syntax errors. Also remove the backticks and the sql keyword."
    "write queries in triple quotes and plain text."
    "Try not to use clause and limit in the SQL."
    "Write your SQL queries in plain text without any Markdown formatting, such as backticks or sql tags."
)

# Create a Tool object for customer data with the previously defined agent executor 'create_sql_agent' and description
mmmVolume_data_tool = Tool(
    name="MMMVolume",
    func=agent_executor.run,
    description=description,
)



# Create a Tool object to grab the data for building the models
mmmModel_data_tool = Tool(
    name="ModelData",
    func=agent_executor.run,
    description=description,
)





# Create the whole list of tools
tools = [PythonREPLTool()]
tools.append(mmmVolume_data_tool)
tools.append(mmmModel_data_tool)


# Define the prefix and suffix for the prompt
prefix = "Below are tools that you can access:"
suffix = (
    "Pass the relevant part of the request directly to the MMM tool.\n\n"
    "Request: {input}\n"
    "{agent_scratchpad}"
)

# Create the prompt using ZeroShotAgent
# Use agent_scratchpad to store the actions previously used, guiding the subsequent responses.
agent_prompt = ZeroShotAgent.create_prompt(
    tools, prefix=prefix, suffix=suffix, input_variables=["input", "chat_history","agent_scratchpad"]
)

# Create an instance of ZeroShotAgent with the LLMChain and the allowed tool names
zero_shot_agent = ZeroShotAgent(
    llm_chain=LLMChain(llm=llm, prompt=agent_prompt),
    allowed_tools=[tool.name for tool in tools]
)

# Initiate memory which allows for storing and extracting messages
memory = ConversationBufferMemory(memory_key="chat_history")


# Create an AgentExecutor which enables verbose mode and handling parsing errors
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=zero_shot_agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory
)

#--------------- begin test for model object --------

# Load the pre-trained model from the .pkl file
fitted_models = {}

for model_name, file_path in model_files.items():
    print(str(model_name), file_path)
    with open(file_path, 'rb') as file:
        fitted_models[model_name] = dill.load(file) #pickle
        print(f"{model_name}: {type(fitted_models[model_name])}")




#@tool(parse_docstring=True)
# Define a function to query the model
def query_model(input_data):

    """
    Query the model with input variables and provide a forecast output.

    Args:
        input_data (list): The input data for prediction.

    Returns:
        list: The prediction results.
    """
    predictions = pd.DataFrame()
    #prediction = fitted_model.predict([input_data], uncertainty=0.05)
    predictions['lb'] = fitted_models['ml_model_obj_lower'].predict([input_data], [input_data])#[0]
    predictions['mid'] = fitted_models['ml_model_obj_mid'].predict([input_data], [input_data])#[0]
    predictions['ub'] = fitted_models['ml_model_obj_upper'].predict([input_data], [input_data])#[0]
    
    return print(f"{predictions}: {(predictions[model_name])}")

    

# Initialize LangChain with your query function
model_tool = Tool(name="query_model", func = query_model,
description="This tool queries the machine learning model with input data and returns the prediction."
)

#--------------- end test for model object --------

# Define the background information
background_info = """
As the customer analyst, my role is to analyze data of advertisement channels and marketing mix models. The feature engineering in table 'mmmVolumeTable' is crucial for statistical exploration. 
For example: - column 'date' can be grouped into bins of weeks and months ranges, such as - Week 2023-01: 2127 - Week 2023-02: 2516 and so on. 
Understanding the data in these columns helps us gain insights about our advertisement, 
enabling us to offer personalized services and develop effective marketing strategies for marketing the drug.
Use all data at all times.
Use all columns of data when asked to summarize output.
"""




#user_inquiry = "Tell me about the data trends for soa engagement and impression for all data. Group by deciles. Aggregate data at the monthly level" 
#Aggregate data at monthly level. Create a trend line and scatter plot to visualize the result of the following inquiry using the python agent \
#Ensure plot is well formatted including dates and characters "



# Define user input
user_inquiry = """
Query the row of data from mmmModelTable using these columns where date is 2/1/2022. That is "select * from table limit 1". Use the list of columns below: 
[decile_onehot__decile_1, decile_onehot__decile_2, decile_onehot__decile_3,decile_onehot__decile_4,
decile_onehot__decile_5,	decile_onehot__decile_6,decile_onehot__decile_7,decile_onehot__decile_8,
decile_onehot__decile_9,	decile_onehot__decile_10,target_segment_onehot__target_segment_CRITICAL,
target_segment_onehot__target_segment_HIGH,target_segment_onehot__target_segment_LOW,
target_segment_onehot__target_segment_MEDIUM,target_segment_onehot__target_segment_NOT_TARGET,
target_segment_onehot__target_segment_PROSPECTS,specialty_onehot__specialty_ENDO,
specialty_onehot__specialty_OTHER,specialty_onehot__specialty_PCP,year_onehot__year_2023,
year_onehot__year_2024,num__details_4,num__details_8,num__details_12,num__samples_4,
num__samples_8,num__samples_12,	num__vae_4,	num__vae_8,	num__vae_12,
num__p2p_4,num__p2p_8,num__p2p_12,num__soa_engagement_8,num__soa_engagement_12,
num__emails_hq_4,	num__emails_hq_8,num__emails_hq_12].

Take the output of the query and increase the numeric data for 'num__p2p_12','num__soa_engagement_8', and 'num__soa_engagement_12'
by 10 percent of their value as the new dataframe and proceed to the next step by the python agent. Use all data from the sql agent.
Use all 38 variables for prediction. 
Load the trained models using the function provided:


for model_name, file_path in model_files.items():
    print(str(model_name), file_path)
    with open(file_path, 'rb') as file:
        fitted_models[model_name] = dill.load(file) #pickle

the model files are shown below        

model_files = {
    "ml_model_obj_lower": "/Users/L052033/Downloads/model.dill", #model.dill
    "ml_model_obj_mid": "/Users/L052033/Downloads/model.dill",
    "ml_model_obj_upper": "/Users/L052033/Downloads/model.dill"
}

as ml_model_obj_lower, ml_model_obj_mid, ml_model_obj_upper.
Use the python agent to predict the forecast with the query_model function: 

predictions['lb'] = fitted_models['ml_model_obj_lower'].predict([input_data],[input_data])
predictions['mid'] = fitted_models['ml_model_obj_mid'].predict([input_data],[input_data])
predictions['ub'] = fitted_models['ml_model_obj_upper'].predict([input_data],[input_data])

Limit the result to one. Use the sklearn predict function to compute the actual value.
"""
#tell me about how nbrx is trend over time with a plot/ visualization
#Load the trained model 'model_tst.pkl' as ml_model_obj. 

agent_executor.run(template.format(background_info=background_info, user_inquiry=user_inquiry))

# Define follow-up question as user input
#user_inquiry = "Can you elaborate more?"
