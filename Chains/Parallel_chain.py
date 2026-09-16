# we shall build a application
#model1->creating notes
#model2->creating quiz on the notes
# model3->adding both and returing 


from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation")

model1=ChatHuggingFace(llm=llm)

model2=ChatHuggingFace(llm=llm)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template="Generate a short and simple notes on \n{topic}",
    input_variables=['topic'])


prompt2=PromptTemplate(
    template="Generate a short and simple notes on \n{topic}",
    input_variables=['topic'])

prompt3=PromptTemplate(
    template="Merge the following notes and quiz into the Single Document By keeping an Relative Heading\n{notes}{quiz}",
    input_variables=['notes','quiz'])

Parallel_chain=RunnableParallel({
    "notes":prompt1| model1 |parser,
    "quiz":prompt2 | model2| parser})


# Parallel_chain.invoke({'text':text})
merge_chain=prompt3|model1| parser

chain=Parallel_chain|merge_chain
text=""" 
Introduction: Why LangChain Matters in a GenAI Interview 
Large Language Models are incredibly powerful, but out of the box, they are isolated. They lack access to real-time data, have limited context windows, and struggle with multi-step reasoning. In a GenAI interview, LangChain should be introduced as the ultimate orchestration framework that bridges these gaps. It allows developers to chain together prompts, models, and external tools to build context-aware, reasoning applications. Showing a deep understanding of LangChain signals to an interviewer that you can transform static LLMs into dynamic, enterprise-grade agents. 
Core Architectural Pillars to Discuss 
When asked how you build GenAI applications, break down your LangChain expertise into its core components. This demonstrates a structured approach to system design: 
1. Models and Prompts (The Foundation) • Concept: LangChain provides a unified interface to interact with various LLMs (e.g., OpenAI, Anthropic, Hugging Face) and manage prompt templates. 
• Interview Talking Point: Discuss  and how they separate prompt logic from application code. Highlight your ability to seamlessly swap underlying models with minimal code changes. 

2. Chains and LCEL (The Workflow) • Concept: LangChain Expression Language (LCEL) is a declarative way to compose chains. 
• Interview Talking Point: Emphasize LCEL. Explain how it natively supports streaming, asynchronous execution, and parallel steps. Instead of just saying "I use chains," explain how  and  optimize data flow. 

3. Data Connection (Retrieval Augmented Generation - RAG) • Concept: Standard LLMs suffer from hallucinations and lack proprietary data. RAG solves this by fetching relevant documents to inform the model's response. 
• Interview Talking Point: Walk the interviewer through a production RAG pipeline: 

	• Loading & Splitting: Using  and text splitters (like ) optimized by chunk size and overlap. 
	• Vector Stores: Embedding text via embedding models and storing them in vector databases (e.g., Pinecone, Chroma, pgvector). 
	• Retrieval: Moving beyond basic similarity search to advanced techniques like Parent Document Retrieval or Contextual Compression to reduce noise. 

4. Memory (Maintaining State) • Concept: LLMs are stateless. Memory components allow applications to remember past interactions. 
• Interview Talking Point: Differentiate between memory types based on use cases. Explain when to use  (short chats) versus  (long conversations to save token costs). 

5. Agents (Autonomous Action) • Concept: Agents use an LLM as a reasoning engine to determine which actions to take and which tools to use. 
• Interview Talking Point: Explain the ReAct (Reason + Action) framework. Describe how you equip an agent with   (e.g., a  Google Search tool  or a database executor) so it can solve complex, non-linear problems autonomously. 

Addressing Production Challenges (The "Senior" Answers) 
Interviewers love to ask, "What happens when things go wrong in production?" Use LangChain concepts to answer these advanced questions: 

• Cost and Latency Optimization: Explain how you use LangChain’s built-in caching ( or ) to avoid calling the LLM API for identical queries, saving money and reducing response times. 
• Debugging and Observability: Mention LangSmith. Explain how you use it to trace chain execution, identify which step caused a failure, inspect exact prompt inputs/outputs, and monitor token consumption. 
• Fallback Strategies: Discuss implementing  in LCEL so that if a primary model (like GPT-4) hits a rate limit or goes down, the system automatically routes the request to a backup model (like Claude or a local Llama model). 

Conclusion 
In a GenAI interview, LangChain is your vehicle to prove you are a practical engineer, not just an AI enthusiast. By articulating how you use LCEL for efficient workflows, RAG for data grounding, and LangSmith for production observability, you position yourself as a candidate ready to build scalable, resilient, and intelligent enterprise systems. 




"""
# result=chain.invoke({'topic':text})
# result=Parallel_chain.invoke({'topic':text})
print(result['notes'])

chain.get_graph().print_ascii()

