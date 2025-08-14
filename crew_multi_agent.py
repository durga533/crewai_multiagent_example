from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv('GOOGLE_API_KEY')

LLM = LLM(model="gemini/gemini-2.0-flash", api_key= key )

news_generator = Agent(
    name = "News agent",
    role = "News gathering agent",
    goal = "You are specialised agent who can get the top 10 news across the world",
    backstory="You are agent who can get the news across the world for consumption",
    tool = SerperDevTool(),
    llm= LLM
)

task1 = Task(

            agent = news_generator,
            name = "Post Generation",
            description = "Get top 10 news across the world",
            expected_output= " Top news across world for human consumption"
)

summarizer= Agent(
    name = "Summarizer agent",
    role = "Summarizer agent",
    goal = "You are specialised agent who can summarize the news gathered into crisp yet useful news lines for consumption",
    backstory="You are agent who can take news generated from previous task and summarize it for quick consumption",
    llm= LLM
)

task2 = Task(

            agent = summarizer,
            name = "Summary generator",
            description = "summarize the top 10 news that was generated from previous task",
            expected_output= " Crisp top 10 news bullets for human consumption"
)

crew = Crew(

            agents= [news_generator, summarizer],
            tasks= [task1, task2],
            verbose = False,
            process="sequential"
)

print(crew.kickoff())




