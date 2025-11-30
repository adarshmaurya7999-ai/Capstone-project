# Capstone-project
This project is an AI Data Analyst Agent that automates the full data workflow using Gemini Function Calling. It plans, cleans data, performs analysis, and generates comprehensive reports.
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
# Imports for Gemini SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

os.environ['GEMINI_API_KEY'] = 'AIzaSyByJsFD_ExytXS1F12xaUmtEyvxSN1U_o0'

# --- 0. Setup: Initialize Gemini Client ---
try:
    if 'GEMINI_API_KEY' not in os.environ:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
    client = genai.Client()
    print("Gemini Client initialized successfully.")
    
except (ValueError, APIError, Exception) as e:
    print(f"FATAL WARNING: Gemini Client failed to initialize. Error: {e}")
    print("Please set your API key. The agent cannot proceed with its task.")
    client = None

try:
    # Initialize the client. This assumes the GEMINI_API_KEY environment variable is set.
    client = genai.Client()
    print("Gemini Client initialized successfully.")
except Exception:
    print("WARNING: Gemini Client failed to initialize. The code will execute Mock Mode.")
    client = None

# --- 1. Custom Tools (Data Analyst Specific) ---

class DataCleaningTool:
    """A custom tool for automated data cleaning and preprocessing."""
    def clean_data(self, df_json: str) -> str:
        """
        Detects missing values, converts types, and removes duplicates.
        Returns the cleaned DataFrame as a JSON string and a summary of actions.
        """
        try:
            df = pd.read_json(df_json)
        except Exception:
            return "Error: Could not parse the input data as a valid JSON DataFrame."

        initial_rows = len(df)
        missing_report = df.isnull().sum()
        
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                elif df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        df.drop_duplicates(inplace=True)
        
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
            
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass

        df['Revenue'].fillna(df['Revenue'].median(), inplace=True)
        
        summary = (f"Data Cleaning Summary:\n"
                   f"- Initial Rows: {initial_rows}, Final Rows: {len(df)}\n"
                   f"- {initial_rows - len(df)} duplicates removed.\n"
                   f"- Missing values handled (imputed with median/mode) in columns:\n"
                   f"  {missing_report[missing_report > 0].to_dict()}")
                   
        return json.dumps({"data": df.to_json(orient='records'), "summary": summary})

class EDAAndVisualizationTool:
    """A custom tool for generating descriptive stats and visualizations."""
    
    def _generate_plot(self, df: pd.DataFrame, filename: str) -> str:
        """Generates a Time-Series Plot for 'Revenue' over 'Date'."""
        if 'Revenue' in df.columns and 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.sort_values('Date').dropna(subset=['Date', 'Revenue'])

                plt.figure(figsize=(10, 5))
                plt.plot(df['Date'], df['Revenue'], marker='o', linestyle='-', color='skyblue')
                plt.title('Time-Series Analysis of Revenue')
                plt.xlabel('Date')
                plt.ylabel('Revenue')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(filename)
                plt.close()
                return filename
            except Exception as e:
                return f"Error generating plot: {e}"
        return ""

    def analyze_data(self, df_json: str) -> str:
        """Generates descriptive statistics and a visualization."""
        try:
            df = pd.read_json(df_json, orient='records', convert_dates=['Date']) 
        except Exception:
            return "Error: Could not parse the input data as a valid JSON DataFrame."

        stats = df.describe(include='all').to_json()

        numeric_cols = df.select_dtypes(include=['number']).columns
        correlation = ""
        if len(numeric_cols) >= 2:
            correlation_series = df[numeric_cols].corr().unstack().sort_values(ascending=False).drop_duplicates()
            correlation = correlation_series[(correlation_series != 1) & (abs(correlation_series) > 0.5)].to_json()
        
        plot_file = ""
        plot_suggestion = ""
        if 'Revenue' in df.columns and 'Date' in df.columns:
            plot_suggestion = "Time-series plot of Revenue over Date."
            plot_file = self._generate_plot(df, "analysis_plot.png")
            
        return json.dumps({
            "descriptive_stats": json.loads(stats),
            "high_correlations": json.loads(correlation) if correlation else "No strong correlations found.",
            "visual_suggestion": plot_suggestion,
            "plot_filename": plot_file
        })

# --- 2.1 Tool for Reporting ---

class ReportingTool:
    """Synthesizes structured data into a natural language report."""
    def generate_report(self, structured_json: str) -> str:
        """Takes the results from EDA and generates a formatted report."""
        try:
            results = json.loads(structured_json)
        except Exception:
            return "Error: Could not parse the structured analysis data."

        stats = results.get('descriptive_stats', {})
        corr = results.get('high_correlations', "None found.")
        plot_file = results.get('plot_filename', "No plot generated.")

        revenue_stats = stats.get('Revenue', {})
        
        report = []
        report.append("## 📈 Data Analysis Final Report: Transaction Data")
        report.append("---")
        
        if results.get('cleaning_summary'):
             report.append("### 🧹 Data Preparation")
             report.append(results['cleaning_summary'])
             report.append("")
        
        report.append("### 📊 Key Performance Indicators (KPIs)")
        if revenue_stats:
            report.append(f"* **Total Transactions:** {stats.get('TransactionID', {}).get('count', 'N/A')}")
            report.append(f"* **Average Revenue:** ${float(revenue_stats.get('mean', 0)):.2f}")
            report.append(f"* **Maximum Revenue:** ${float(revenue_stats.get('max', 0)):.2f}")
            report.append(f"* **Minimum Revenue:** ${float(revenue_stats.get('min', 0)):.2f}")
        
        report.append("\n### 🔍 Deep Dive Analysis (Insights)")
        
        max_rev = float(revenue_stats.get('max', 0))
        report.append(f"The analysis shows a peak revenue of **${max_rev:.2f}**. This is a key data point for further investigation.")
        
        if corr != "No strong correlations found.":
            report.append(f"**Strong Correlations Found:** {corr}. Further regression analysis is recommended.")
        else:
            report.append("No strong linear correlations (R > 0.5) were found between numeric variables.")
            
        report.append("\n### 🖼️ Revenue Time-Series Visualization")
        
        if plot_file and "Error" not in plot_file:
            report.append(f"""The time-series plot below illustrates the revenue trend over the analyzed period. 



[Image of analysis_plot.png]

""")
        else:
            report.append("A time-series plot could not be generated due to data limitations or error.")
            
        return "\n".join(report)


class GeminiAgent:
    """
    A unified agent that uses the Gemini SDK for dynamic reasoning and tool execution.
    This replaces the MockLLM, the Agent base class, and the Orchestrator.
    """
    def __init__(self, client):
        self.client = client
        self.name = "Gemini Data Analyst Agent"
        
        self.cleaning_tool = DataCleaningTool()
        self.eda_tool = EDAAndVisualizationTool()
        self.reporting_tool = ReportingTool()
        
        self.tools_map = {
            "clean_data": self.cleaning_tool.clean_data,
            "analyze_data": self.eda_tool.analyze_data,
            "generate_report": self.reporting_tool.generate_report
        }
        
        self.tool_definitions = list(self.tools_map.values())

    def run_workflow(self, user_task: str, input_json: str) -> str:
        """
        Executes a multi-step workflow driven by Gemini's tool calling ability.
        """
        if not self.client:
            return "Error: Gemini Client is not initialized. Cannot run workflow. Please set GEMINI_API_KEY."

        system_instruction = ("You are a world-class Data Analyst Agent. Your role is to plan the "
                              "complete workflow: Data Cleaning, Exploratory Analysis, and final Reporting. "
                              "You must call the 'clean_data' tool first, then chain the result to the "
                              "'analyze_data' tool, and finally use the structured analysis output to "
                              "call 'generate_report'. Do not generate the report text yourself; let the tools execute.")
        
        initial_prompt = (
            f"The raw transaction data is provided below as a JSON string. Perform the following task:\n\n"
            f"RAW DATA: {input_json}\n\n"
            f"PRIMARY USER TASK: {user_task}"
        )
        
        history = [
            types.Content(
                role="user", 
                parts=[types.Part.from_text(initial_prompt)]
            )
        ]
        
        # Start the generation loop
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=history,
                tools=self.tool_definitions,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )
            
            # Context dictionary to pass data between tool calls
            workflow_context = {"raw_input": input_json}
            
            while response.function_calls:
                
                function_calls_parts = []
                function_results_parts = []
                
                for function_call in response.function_calls:
                    func_name = function_call.name
                    
                    if func_name == "clean_data":
                        tool_input = workflow_context["raw_input"]
                    elif func_name == "analyze_data":
                        tool_input = workflow_context.get("cleaned_data_json", "{}") 
                    elif func_name == "generate_report":
                        tool_input = workflow_context.get("analysis_output_json", "{}")
                    else:
                        tool_input = json.dumps(dict(function_call.args))

                    print(f"\n[AGENT ACTION]: Calling {func_name}...")
                    
                    # Execute the Tool
                    tool_output_str = self.tools_map[func_name](tool_input)
                    
                    # Store results for the next tool call and final report
                    if func_name == "clean_data":
                        cleaning_output = json.loads(tool_output_str)
                        workflow_context["cleaned_data_json"] = cleaning_output["data"]
                        workflow_context["cleaning_summary"] = cleaning_output["summary"]
                    elif func_name == "analyze_data":
                        analysis_output = json.loads(tool_output_str)
                        final_report_input = {
                            'cleaning_summary': workflow_context.get("cleaning_summary", "N/A"),
                            **analysis_output
                        }
                        workflow_context["analysis_output_json"] = json.dumps(final_report_input)

                    function_results_parts.append(
                        types.Part.from_function_response(
                            name=func_name,
                            response={"result": tool_output_str}
                        )
                    )
                    function_calls_parts.append(types.Part.from_function_call(function_call))

                history.append(types.Content(role="model", parts=function_calls_parts))
                history.append(types.Content(role="tool", parts=function_results_parts))

                # Next Call: LLM plans the subsequent step
                print("[LLM THINKING]: Receiving tool results and determining the next step...")
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=history,
                    tools=self.tool_definitions,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    )
                )
                
            # Final output when LLM is done calling tools and generates the final text
            return response.text

        except APIError as e:
            return f"Gemini API Error: {e}"
        except Exception as e:
            return f"General Execution Error: {e}"

# --- 4. Workflow Execution (Modified for GeminiAgent) ---

mock_data = {
    'TransactionID': [1, 2, 3, 4, 5, 5, 6],
    'Date': ['2023-10-01', '2023-10-05', '2023-10-10', '2023-10-15', '2023-10-20', '2023-10-20', 'Invalid_Date'],
    'Product': ['A', 'B', 'A', 'C', 'B', 'B', 'A'],
    'Revenue': [100.0, 150.5, None, 200.0, 150.5, 150.5, 90.0],
    'Region': ['East', 'West', 'East', 'Central', 'West', 'West', None]
}
input_df = pd.DataFrame(mock_data)
input_json = input_df.to_json(orient='records')


print(f"**--- Starting Gemini Data Analysis Workflow at {datetime.now().strftime('%H:%M:%S')} ---**")

gemini_agent = GeminiAgent(client)

user_request = "Perform a complete end-to-end analysis on the provided transaction data. Clean the data, conduct an exploratory analysis, and generate a final comprehensive report."

# The single call that triggers the entire workflow (Clean -> EDA -> Report)
final_report = gemini_agent.run_workflow(user_request, input_json)

print("\n\n*** FINAL GEMINI-GENERATED REPORT ***")
print(final_report)

print("\n**--- Workflow Complete ---**")
