from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate


load_dotenv()

prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation")

model=ChatHuggingFace(llm=llm)

prompt2= PromptTemplate(
    template="Generate a summary in 2 lines about the report{report}",
    input_variables=['report'])

parser=StrOutputParser()

chain=prompt1 | model |parser | prompt2 | model | parser

result=chain.invoke({"topic":"IndianFlag"})

print(result)