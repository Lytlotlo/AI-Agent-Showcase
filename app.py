from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
import os

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

@tool
def get_ai_safety_updates(category: str = "general", max_results: int = 5) -> str:
    """
    A tool that fetches the latest news updates on AI safety developments and categorizes them.
    
    Args:
        category: The topic or category for the news updates (e.g., 'research', 'policy', 'industry').
        max_results: The maximum number of news articles to retrieve.
    
    Returns:
        A string summarizing the latest AI safety news updates for the specified category.
    """

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return "Error: NEWS_API_KEY is not set in your environment."

    # Prepare the query; you can customize it further as needed
    query = f" Latest AI safety research News {category}"
    
    # News API endpoint
    url = "https://newsapi.org/v2/everything"
    
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "apiKey": api_key
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        if not articles:
            return f"No recent updates found for category '{category}'."
        
        # Build a summary of the news articles
        summary = f"Latest AI safety updates for category '{category}':\n"
        for article in articles:
            title = article.get("title", "No title")
            source = article.get("source", {}).get("name", "Unknown source")
            published = article.get("publishedAt", "No date")
            summary += f"- {title} (Source: {source}, Date: {published})\n"
        return summary
    else:
        return f"Error fetching news: {response.status_code} - {response.text}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)
model.last_input_token_count = 0


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer,
          get_current_time_in_timezone,
          get_ai_safety_updates], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()