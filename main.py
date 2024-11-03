from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import pandas as pd
import os
from dotenv import load_dotenv
from io import StringIO
import json
import openai
import logging
import numpy as np  # Added import for numpy
import re  # Added import for regular expressions
from typing import Optional
import sys
from io import StringIO
from datetime import datetime



# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()

# print msg in red, accept multiple strings like print statement
def print_red(*strings):
    print("\033[91m" + " ".join(strings) + "\033[0m")


# print msg in blue, , accept multiple strings like print statement
def print_blue(*strings):
    print("\033[94m" + " ".join(strings) + "\033[0m")

async def format_analysis_result(result: str) -> str:
    """
    Use AI to format the analysis result in a more natural, conversational way when appropriate.
    
    Args:
        result (str): Raw analysis result from pandas/python execution
        
    Returns:
        str: Formatted result
    """
    try:
        format_prompt = f"""Format the following data analysis result in a clear and concise way.

        Original result:
        {result}

        Requirements:
        1. For simple queries (like "what is the highest MPG?"), provide a direct, conversational answer
        2. For more complex analyses, use minimal Markdown formatting:
           - Use ## for section headers if needed
           - Use * or - for bullet points when listing multiple items
           - Use ** for important numbers or key findings
           - Use proper line breaks
        3. DO NOT include:
           - Analysis completion timestamps
           - Top level (# level) headers
        4. Keep it concise and natural - if the answer can be expressed in one clear sentence, do that instead of using formal formatting
        
        Example for simple query:
        "The highest MPG is 46.6, recorded for the Toyota Corolla."
        
        Example for complex analysis:
        ## MPG Analysis by Origin
        * European cars: **27.9** mpg average
        * Japanese cars: **30.4** mpg average
        * US cars: **25.2** mpg average"""
        
        messages = [
            {
                "role": "system", 
                "content": """You are a data presentation expert that makes analysis results clear and accessible.
                For simple queries, give direct answers.
                For complex analyses, use minimal formatting to organize information clearly."""
            },
            {"role": "user", "content": format_prompt}
        ]
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        
        
        formatted_result = response.choices[0].message.content
        formatted_result = formatted_result.replace('\r\n', '\n')
        
        return formatted_result

    except Exception as e:
        logging.error(f"Error formatting result: {str(e)}")
        return result  # Return original result if formatting fails

async def generate_data_analysis(query: str, df: pd.DataFrame = None) -> dict:
    """
    Generate and execute data analysis code based on user query.
    
    Args:
        query (str): User's analysis query
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        dict: Contains either analysis result or error message
    """
    if df is None or df.empty:
        raise ValueError("No dataset available")

    try:
        # Get dataset information
        column_info = {
            "columns": list(df.columns),
            "sample": df.head(1).to_dict('records')[0]
        }
        
        # Generate analysis code with dataset context
        analysis_prompt = f"""You are a Python code generator. You have access to a pandas DataFrame with these exact columns:
        {column_info['columns']}
        
        Sample data:
        {column_info['sample']}
        
        Generate Python pandas code to analyze this query: {query}
        
        Important:
        1. The DataFrame is named 'uploaded_df'
        2. Use the EXACT column names as shown above
        3. Generate print statements that include:
           - Clear context about what is being calculated
           - The numerical results with appropriate labels
           - Any relevant additional insights
        
        Example format:
        print("Analysis of MPG by Origin:")
        print("-------------------------")
        print(f"Average MPG:")
        result = uploaded_df.groupby('Origin')['MPG'].mean()
        print(result)
        """
        
        messages = [
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": query}
        ]
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        
        code = response.choices[0].message.content
        raw_result = execute_pandas_dataframe_code(code)
        
        # Format the result to be more human-readable
        formatted_result = await format_analysis_result(raw_result)
        
        return {"response": formatted_result}

    except Exception as e:
        raise Exception(f"Error generating analysis: {str(e)}")
    
async def generate_visualization(prompt: str, df: pd.DataFrame = None) -> dict:
    """
    Generate a Vega-Lite visualization specification with debug logging and data type information.
    """
    print_red("=== Visualization Debug ===")
    
    if df is None or df.empty:
        raise ValueError("No dataset available")

    try:
        # Get column information and types
        columns = df.columns.tolist()
        
        # Determine data types for each column
        types = {}
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                types[col] = "quantitative"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                types[col] = "temporal"
            elif len(df[col].unique()) < len(df[col]) * 0.05:  # If unique values are less than 5% of total rows
                types[col] = "nominal"
            else:
                types[col] = "ordinal"

        # Get sample values
        sample_values = df.head(3).to_dict('records')

        # Debug logging
        print_red(f"DataFrame shape: {df.shape}")
        print_red(f"Columns and types: {types}")
        print_red(f"Sample values: {sample_values}")
        print_red(f"Original prompt: {prompt}")

        # Create enhanced system prompt with data context
        system_prompt = f"""You are a data visualization assistant responsible for generating Vega-Lite specifications.

        Dataset Information:
        - Columns (case sensitive): {columns}
        - Data types for each column: {json.dumps(types, indent=2)}
        - Sample values: {json.dumps(sample_values, indent=2)}

        Requirements:
        1. Use the exact column names shown above in all field specifications
        2. Use "data": {{"values": "myData"}} as the data placeholder
        3. Use proper Vega-Lite aggregation syntax:
           - For counting records: use "aggregate": "count"
           - For other aggregations: use "aggregate": "sum"/"mean"/"median" etc.
           - Never use "field": "count()" or similar SQL-style syntax
        4. Match the data types specified above in your encoding
        5. ONLY provide the JSON specification, no additional text
        
        Example for histogram with count:
        {{
          "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
          "data": {{"values": "myData"}},
          "mark": "bar",
          "encoding": {{
            "x": {{"field": "MPG", "type": "quantitative", "bin": true}},
            "y": {{"aggregate": "count", "type": "quantitative"}}
          }}
        }}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        chat_completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=1000,
        )

        response_text = chat_completion.choices[0].message.content.strip()
        print_red(f"Generated spec:\n{response_text}")
        
        # Validate the specification
        try:
            vega_spec = json.loads(response_text)
            # Basic validation of aggregation syntax
            if 'encoding' in vega_spec:
                for axis, encoding in vega_spec['encoding'].items():
                    if encoding.get('field') == 'count()':
                        print_red("Found invalid count() syntax, fixing...")
                        encoding.pop('field')
                        encoding['aggregate'] = 'count'
            return {"vegaSpec": vega_spec}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON specification: {str(e)}")

    except Exception as e:
        print_red(f"Error in generate_visualization: {str(e)}")
        raise Exception(f"Error generating visualization: {str(e)}")

def sanitize_input(query: str) -> str:
    """
    Sanitize input to the python REPL.
    Removes whitespace, backticks & 'python' prefix from the query.
    
    Args:
        query (str): The input query to sanitize
    
    Returns:
        str: Sanitized query
    """
    # Remove leading whitespace, backticks, and 'python' prefix
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Remove trailing whitespace and backticks
    query = re.sub(r"(\s|`)*$", "", query)
    return query

def execute_pandas_dataframe_code(code: str) -> str:
    """
    Execute the given python code that operates on a pandas DataFrame and return the output.
    
    Args:
        code (str): Python code to execute (must use print() for output)
    
    Returns:
        str: Output from the code execution or error message
    """
    # Save the current standard output
    old_stdout = sys.stdout
    # Redirect standard output to capture any printed output
    sys.stdout = mystdout = StringIO()
    
    try:
        # Clean and execute the code
        cleaned_code = sanitize_input(code)
        # Check if the code attempts to access global uploaded_df
        if 'uploaded_df' not in cleaned_code:
            cleaned_code = cleaned_code.replace('df', 'uploaded_df')
        
        # Add error handling for empty DataFrame
        prefix_code = """
if uploaded_df is None or uploaded_df.empty:
    print("Error: No data available. Please upload a dataset first.")
else:
"""
        # Indent the user code
        indented_code = "\n".join("    " + line for line in cleaned_code.split("\n"))
        final_code = prefix_code + indented_code
        
        # Execute the code
        exec(final_code, {'uploaded_df': uploaded_df, 'pd': pd, 'np': np})
        output = mystdout.getvalue().strip()
        return output if output else "Operation completed successfully but produced no output. Make sure to use print() to show results."
        
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        # Restore the standard output
        sys.stdout = old_stdout

# Define tools based on existing functions in main.py
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": """This function generates a Vega-Lite visualization based on the provided prompt.
            Use this when the user needs:
            - Any type of chart or graph visualization
            - Visual data comparisons
            - Distribution plots or histograms
            - Scatter plots or correlation views
            The function will return a complete Vega-Lite specification.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The Vega-Lite visualization prompt with data context"
                    },
                    "df": {
                        "type": "object",
                        "description": "DataFrame to verify data existence"
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_data_analysis",
            "description": """This function generates and executes pandas analysis code based on user query.
            Use this when the user needs:
            - Statistical calculations (mean, median, etc.)
            - Data summaries or aggregations
            - Specific numerical answers
            - Data filtering or grouping results
            Note: Results will be printed using print() statements.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's analysis query"
                    },
                    "df": {
                        "type": "object",
                        "description": "DataFrame to analyze"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]

# Tool mapping to actual functions
tool_map = {
    'generate_visualization': generate_visualization,
    'generate_data_analysis': generate_data_analysis
}


app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the uploaded dataset
uploaded_df = None

# Define request and response models
class QueryRequest(BaseModel):
    prompt: str
    userQuery: str  # Added field to receive the user's original input

class QueryResponse(BaseModel):
    response: Optional[str] = None  # Make the response field optional
    vegaSpec: Optional[dict] = None  # Make the vegaSpec field optional as well if needed

    
# Endpoint to handle file upload
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_df
    try:
        content = await file.read()
        uploaded_df = pd.read_csv(StringIO(content.decode('utf-8')))
        return {"status": "success", "message": "File uploaded and parsed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload and parse file: {str(e)}")

# Endpoint to provide a preview of the uploaded dataset
@app.get("/preview")
async def preview_data():
    global uploaded_df

    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    try:
        preview_data = uploaded_df.head(10).to_dict(orient='records')  # Return first 5-10 rows as a preview
        return {"status": "success", "preview": preview_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found in environment variables.")

# Endpoint to handle user queries
# Endpoint to handle user queries
@app.post("/query", response_model=QueryResponse)
async def query_openai(request: QueryRequest):
    """
    Process user queries using ReAct logic with OpenAI function calling.
    Shows thought process through detailed logging.
    """
    global uploaded_df
    
    print("\n=== New Query Started ===")
    print_red(f"User Query: {request.userQuery}")

    if uploaded_df is None:
        print_red("Thought: No dataset available. Must inform user to upload first.")
        return QueryResponse(response="Please upload a dataset first.")
        
    try:
        # Define the ReAct loop system prompt
        system_prompt = """You are a helpful assistant that can perform data analysis and visualization based on user requests.
You have access to the following tools:
1. generate_data_analysis: For statistical calculations, summaries, or numerical answers
2. generate_visualization: For charts, graphs, and visual data representations

For each response, follow this process:
1. Think about what type of analysis or visualization would best answer the user's question
2. Choose the appropriate tool based on your thinking
3. Execute the tool and observe results
4. Provide a clear explanation to the user

Example thought process:
User: "What's the average MPG for European cars?"
Thought: This requires a statistical calculation on filtered data
Action: Use generate_data_analysis
Observation: [Result from analysis]
Final Answer: [Clear explanation with the result]"""

        print_blue("System prompt set. Starting ReAct loop...")
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": request.userQuery})
        
        max_iterations = 5
        iteration = 0
        final_result = {}

        while iteration < max_iterations:
            iteration += 1
            print_red(f"\n=== Iteration {iteration} Started ===")

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            
            # Print thought process
            if response_message.content:
                print_red(f"Assistant's Thought: {response_message.content}")

            if response_message.content and not response_message.tool_calls:
                print_red("Final Answer reached without tool use")
                return QueryResponse(response=response_message.content)

            if response_message.tool_calls:
                print_blue("Tool calls detected. Processing...")
                messages.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    print_blue(f"Action: Using {function_name}")
                    
                    arguments = json.loads(tool_call.function.arguments)
                    if 'df' not in arguments:
                        arguments['df'] = uploaded_df
                        
                    function_to_call = tool_map[function_name]
                    result = await function_to_call(**arguments)
                    
                    print_blue(f"Observation: Got result from {function_name}")
                    
                    if 'vegaSpec' in result:
                        final_result['vegaSpec'] = result['vegaSpec']
                    if 'response' in result:
                        final_result['response'] = result['response']
                    
                    result_content = json.dumps({
                        **{k: v for k, v in arguments.items() if k != 'df'},
                        "result": result
                    })
                    
                    tool_response_message = {
                        "role": "tool",
                        "content": result_content,
                        "tool_call_id": tool_call.id
                    }
                    messages.append(tool_response_message)

            if len(final_result) > 0:
                print_red("Final result ready to return")
                return QueryResponse(**final_result)

            iteration += 1

        return QueryResponse(response="Unable to complete the request within iteration limit.")

    except Exception as e:
        print_red(f"Error occurred: {str(e)}")
        return QueryResponse(response=f"Error processing query: {str(e)}")
    
# Root endpoint to serve the index.html file
@app.get("/")
async def read_root():
    return FileResponse('static/index.html')