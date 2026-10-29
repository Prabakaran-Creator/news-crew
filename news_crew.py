from datetime import date
from dotenv import load_dotenv

load_dotenv(override=True)  # read keys from the .env file

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

today = date.today().strftime("%d %B %Y")

search = SerperDevTool()

researcher = Agent(
    role="Tech News Researcher",
    goal="Find today's top 3 technology news stories using web search",
    backstory=(
        "You are a sharp tech journalist who checks the web every day. "
        "You only report stories that are recent, real and verified by search results."
    ),
    tools=[search],
    verbose=True,
)

writer = Agent(
    role="Newsletter Writer",
    goal="Turn the researched stories into a clean, exactly 5-line tech newsletter",
    backstory=(
        "You are a concise newsletter editor. You never search the web. "
        "You only use the stories you are given and write them in short, clear lines."
    ),
    verbose=True,
)

find = Task(
    description=(
        f"Today's date is {today}. Search the web for today's top 3 technology "
        f"news stories. Include '{today}' in your search queries. "
        "Only pick stories whose search result clearly shows they were published "
        "today or yesterday. Skip anything with no clear date. "
        "Make sure the 3 stories are different from each other."
    ),
    expected_output=(
        "A numbered list of exactly 3 stories. Each item has a short title "
        "and a one-sentence summary."
    ),
    agent=researcher,
)

write = Task(
    description=(
        "Using only the 3 stories provided, write a short tech newsletter. "
        "Do not search the web or add stories of your own."
    ),
    expected_output=(
        "Exactly 5 lines, no more and no less:\n"
        "Line 1: Tech in 5 lines\n"
        "Lines 2-4: '1.', '2.', '3.' followed by one story, one line each\n"
        "Line 5: Why it matters: followed by one line"
    ),
    agent=writer,
    context=[find],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[find, write],
    process=Process.sequential,
    verbose=True,
)

print(crew.kickoff())