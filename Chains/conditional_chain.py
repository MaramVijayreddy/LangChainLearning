#we shall build a customer_feedback behaviour 


from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation")


class Feedback(BaseModel):
    sentiment :Literal['Positive','Negative']=Field(description='Give the sentiment of the feedback')


parser=PydanticOutputParser(pydantic_object=Feedback)

parser2=StrOutputParser()

prompt1=PromptTemplate(
    template="Analyze the Text into Positive or Negative\n{feedback}{format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
model1=ChatHuggingFace(llm=llm)

model2=ChatHuggingFace(llm=llm)
classifier_chain= prompt1 | model1 |parser

prompt2=PromptTemplate(
    template="Write a Relative Response TO the Following Feedback{feedback}",
    input_variables=['feedback']
)
prompt3 =PromptTemplate(
    template="Give a Appropriate Mail TO the following Feedback{feedback}",
    input_variables=['feedback']
)

branchchain=RunnableBranch(
    (lambda x : x.sentiment =='Positive',prompt2|model1|parser2),
    (lambda x :x.sentiment=='Negative',prompt3|model1|parser2),
    RunnableLambda(lambda x:'Cant Find the Sentiment'))


chain= classifier_chain|branchchain

text="THis is a  phone"

result=chain.invoke({'feedback':text})

print(result)


print(chain.get_graph().print_ascii())

