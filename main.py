import os
import re
import json
import requests
import time
import argparse
import sys
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import geocoder

load_dotenv()

class LLMProvider:
    def __init__(self, model="llama2"):
        self.model = model
        self.base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        print(f"Initialized Ollama LLM provider with model: {self.model}")

    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API request failed with status {response.status_code}: {response.text}")
            
            try:
                response_json = response.json()
                full_response = response_json.get("response", "").strip()
                return full_response
            except json.JSONDecodeError:
                print("WARNING: Received a streaming response despite requesting non-streaming.")
                print("Raw response text (first 500 chars):", response.text[:500])
                
                try:
                    first_line = response.text.split('\n')[0]
                    return json.loads(first_line).get("response", "")
                except:
                    return "Error: Unable to parse Ollama response. Make sure Ollama version is up to date."
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            if 'response' in locals():
                print(f"Response content (first 500 chars): {response.text[:500] if hasattr(response, 'text') else 'No response text'}")
            return f"Error generating response: {str(e)}"


class LocationService:
    def __init__(self):
        self.cache_time = 300  # Cache location for 5 minutes
        self.last_fetch = 0
        self.cached_location = None
    
    def get_current_location(self):
        current_time = time.time()
        
        if self.cached_location and (current_time - self.last_fetch) < self.cache_time:
            return self.cached_location
            
        try:
            g = geocoder.ip('me')
            if g.ok:
                self.cached_location = {
                    'city': g.city,
                    'state': g.state,
                    'country': g.country,
                    'latlng': g.latlng
                }
                self.last_fetch = current_time
                return self.cached_location
            else:
                print(f"Failed to get location: {g.error}")
                return None
        except Exception as e:
            print(f"Error getting location: {str(e)}")
            return None


class Tool:
    def __init__(self, name: str, description: str, function):
        self.name = name
        self.description = description
        self.function = function
    
    def execute(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations",
            function=self.calculate
        )
    
    def calculate(self, expression: str) -> str:
        try:
            safe_dict = {"__builtins__": None}
            safe_math = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
            
            sanitized = re.sub(r'[a-zA-Z_][\w]*', '', expression)
            result = eval(sanitized, {"__builtins__": None}, safe_math)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"


class SearchTool(Tool):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("SERPAPI_KEY")
        
        super().__init__(
            name="search",
            description="Search the web for information",
            function=self.search
        )
    
    def search(self, query: str) -> str:
        try:
            response = requests.get(
                "https://serpapi.com/search",
                params={
                    "api_key": self.api_key,
                    "q": query,
                    "engine": "google"
                }
            )
            data = response.json()
            
            if "error" in data:
                return f"Error: {data['error']}"
            
            results = []
            
            for result in data.get("organic_results", [])[:3]:
                results.append(f"- {result['title']}: {result['snippet']}")
            
            return "\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Failed to search: {str(e)}"


class GoogleMapsSearchTool(Tool):
    def __init__(self, api_key: str = None, location_service=None):
        self.api_key = api_key or os.environ.get("SERPAPI_KEY")
        self.location_service = location_service
        
        super().__init__(
            name="maps_search",
            description="Search for places, businesses, and locations on Google Maps",
            function=self.search_places
        )
    
    def search_places(self, query: str) -> str:
        try:
            # Check if query contains "near me" or similar phrases
            if re.search(r'\b(near me|nearby|close to me|around me)\b', query.lower()):
                location = self.location_service.get_current_location() if self.location_service else None
                
                if location and location['latlng']:
                    lat, lng = location['latlng']
                    location_str = f"{location['city']}, {location['state']}" if location['city'] and location['state'] else None
                    
                    # Replace "near me" with actual location if available
                    if location_str:
                        query = re.sub(r'\b(near me|nearby|close to me|around me)\b', f"in {location_str}", query, flags=re.IGNORECASE)
                    
                    params = {
                        "api_key": self.api_key,
                        "q": query,
                        "engine": "google_maps",
                        "ll": f"@{lat},{lng},15z"  # Set the map center to user's location
                    }
                else:
                    print("Location not available. Using query as-is.")
                    params = {
                        "api_key": self.api_key,
                        "q": query,
                        "engine": "google_maps"
                    }
            else:
                params = {
                    "api_key": self.api_key,
                    "q": query,
                    "engine": "google_maps"
                }
            
            response = requests.get("https://serpapi.com/search", params=params)
            data = response.json()
            
            if "error" in data:
                return f"Error: {data['error']}"
            
            results = []
            
            for place in data.get("local_results", [])[:5]:
                address = place.get("address", "No address available")
                rating = place.get("rating", "No rating")
                
                place_info = f"- {place.get('title')}: {address}"
                if rating != "No rating":
                    place_info += f", Rating: {rating}/5"
                
                results.append(place_info)
            
            if not results:
                return "No places found for that query. Try adding more specific location information like city name."
                
            location_info = ""
            if self.location_service:
                loc = self.location_service.get_current_location()
                if loc:
                    location_info = f"Based on your location in {loc['city']}, {loc['state']}, {loc['country']}:\n\n"
            
            return f"{location_info}{''.join(results)}"
        except Exception as e:
            return f"Failed to search Google Maps: {str(e)}"


class AgentMemory:
    def __init__(self):
        self.short_term = []
        self.context_window = 10
    
    def add(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > self.context_window:
            self.short_term.pop(0)
    
    def get_conversation_history(self) -> str:
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.short_term])


class Agent:
    def __init__(self, llm_provider: LLMProvider, tools: List[Tool] = None):
        self.llm = llm_provider
        self.tools = tools or []
        self.memory = AgentMemory()
        self.max_iterations = 5
        self.verbose = False
    
    def add_tool(self, tool: Tool):
        self.tools.append(tool)
    
    def set_verbose(self, verbose: bool):
        self.verbose = verbose
    
    def _get_system_prompt(self) -> str:
        tools_desc = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        
        return f"""You are an agentic AI assistant. Your goal is to help the user by taking actions autonomously.
You can use these tools:
{tools_desc}

To use a tool, respond in this format:
```
THINKING: Your step-by-step reasoning about what to do
ACTION: tool_name
ACTION_INPUT: the input to the tool
```

If you have a final answer for the user, respond like this:
```
THINKING: Your reasoning
FINAL_ANSWER: Your response to the user
```
"""
    
    def _extract_action(self, llm_response: str) -> Dict[str, str]:
        if self.verbose:
            print(f"\nRaw LLM response:\n{llm_response}\n")
        
        thinking_match = re.search(r"THINKING:(.*?)(?:ACTION:|FINAL_ANSWER:)", llm_response, re.DOTALL)
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        
        action_match = re.search(r"ACTION:(.*?)ACTION_INPUT:", llm_response, re.DOTALL)
        action = action_match.group(1).strip() if action_match else None
        
        action_input_match = re.search(r"ACTION_INPUT:(.*?)(?=```|$)", llm_response, re.DOTALL)
        action_input = action_input_match.group(1).strip() if action_input_match else None
        
        final_answer_match = re.search(r"FINAL_ANSWER:(.*?)(?=```|$)", llm_response, re.DOTALL)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else None
        
        if not thinking and not action and not action_input and not final_answer and llm_response.strip():
            if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', llm_response.lower()):
                for tool in self.tools:
                    if tool.name.lower() == "calculator":
                        calculation_match = re.search(r'[\d\s\+\-\*\/\(\)\.]+', llm_response)
                        if calculation_match:
                            expr = calculation_match.group(0).strip()
                            return {
                                "thinking": f"This appears to be a calculation request: {expr}",
                                "action": "calculator",
                                "action_input": expr,
                                "final_answer": None
                            }
            
            if re.search(r'near me|find|locate|where|places|restaurants|shops|hotels', llm_response.lower()):
                for tool in self.tools:
                    if tool.name.lower() == "maps_search":
                        return {
                            "thinking": f"This appears to be a location search request",
                            "action": "maps_search",
                            "action_input": llm_response,
                            "final_answer": None
                        }
            
            return {
                "thinking": "Processing user request",
                "action": None,
                "action_input": None,
                "final_answer": llm_response.strip()
            }
        
        result = {
            "thinking": thinking,
            "action": action,
            "action_input": action_input,
            "final_answer": final_answer
        }
        
        if self.verbose:
            print("Extracted components:")
            for key, value in result.items():
                if value:
                    print(f"{key}: {value[:100]}..." if len(value) > 100 else f"{key}: {value}")
                else:
                    print(f"{key}: None")
        
        return result
    
    def _execute_action(self, action_name: str, action_input: str) -> str:
        for tool in self.tools:
            if tool.name.lower() == action_name.lower():
                return tool.execute(action_input)
        
        return f"Error: Tool '{action_name}' not found."
    
    def run(self, user_input: str) -> str:
        self.memory.add("user", user_input)
        
        if re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', user_input.strip()):
            for tool in self.tools:
                if tool.name.lower() == "calculator":
                    result = tool.execute(user_input.strip())
                    self.memory.add("assistant", result)
                    return result
        
        if re.search(r'\b(near me|nearby|find|where|around|close to me)\b', user_input.lower()):
            for tool in self.tools:
                if tool.name.lower() == "maps_search":
                    result = tool.execute(user_input.strip())
                    self.memory.add("assistant", result)
                    return f"Here are some places I found:\n\n{result}"
        
        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1
            
            system_prompt = self._get_system_prompt()
            conversation_history = self.memory.get_conversation_history()
            prompt = f"{system_prompt}\n\nConversation history:\n{conversation_history}\n\nWhat should you do next?"
            
            if self.verbose:
                print(f"\nSending prompt to Ollama (iteration {iterations})...")
            
            llm_response = self.llm.generate_response(prompt)
            self.memory.add("assistant_thinking", llm_response)
            
            action_data = self._extract_action(llm_response)
            
            if action_data["final_answer"]:
                self.memory.add("assistant", action_data["final_answer"])
                return action_data["final_answer"]
            
            if action_data["action"] and action_data["action_input"]:
                if self.verbose:
                    print(f"Executing action: {action_data['action']} with input: {action_data['action_input']}")
                
                tool_result = self._execute_action(action_data["action"], action_data["action_input"])
                self.memory.add("system", f"Tool result: {tool_result}")
                
                if self.verbose:
                    print(f"Tool result: {tool_result}")
            else:
                if llm_response.strip():
                    self.memory.add("assistant", llm_response)
                    return llm_response
                    
                self.memory.add("system", "Failed to parse action and input.")
                return "I encountered an error in processing your request. Could you please try again with more details?"
        
        self.memory.add("assistant", "I've hit my iteration limit. Here's my current understanding: " + action_data.get("thinking", ""))
        return "I've spent some time working on this but couldn't complete the task. Here's what I understand so far: " + action_data.get("thinking", "")


def main():
    parser = argparse.ArgumentParser(description="Run an agentic AI demo with Ollama")
    parser.add_argument("--model", default="llama2", help="Ollama model to use (default: llama2)")
    parser.add_argument("--serpapi-key", help="SerpAPI key")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    try:
        location_service = LocationService()
        current_location = location_service.get_current_location()
        if current_location:
            print(f"Location detected: {current_location['city']}, {current_location['state']}, {current_location['country']}")
        else:
            print("Location detection failed. 'Near me' searches will be less accurate.")
        
        llm_provider = LLMProvider(model=args.model)
        agent = Agent(llm_provider)
        agent.set_verbose(args.verbose)
        
        agent.add_tool(CalculatorTool())
        
        serpapi_key = args.serpapi_key or os.environ.get("SERPAPI_KEY")
        if serpapi_key:
            agent.add_tool(SearchTool(api_key=serpapi_key))
            agent.add_tool(GoogleMapsSearchTool(api_key=serpapi_key, location_service=location_service))
            print("Search and Maps tools enabled")
        else:
            print("Search tools disabled (no API key provided)")
            
        print("\nAgentic AI Demo")
        print("---------------")
        print("Using Ollama with model:", llm_provider.model)
        print("Available tools:")
        for tool in agent.tools:
            print(f"- {tool.name}: {tool.description}")
        print("\nType 'exit' to quit.")
        print("---------------")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            response = agent.run(user_input)
            print(f"\nAgent: {response}")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()