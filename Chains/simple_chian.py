from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation")

model=ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="Generate a 5 lines interesting facts about{topic}",
    input_variables=['topic'])

parser=StrOutputParser()

chain= template1 | model |parser

result=chain.invoke({'topic':"India"})
print(result)

chain.get_graph().print_ascii()