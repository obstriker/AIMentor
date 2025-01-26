from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI

class DryRunTokenMonitor(BaseCallbackHandler):
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        if self.dry_run:
            print("Dry Run: Simulating LLM call")
            print("Prompts:", prompts)
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        # Extract token usage from response metadata
        for generation in response.generations:
            usage = generation[[0]](https://python.langchain.com/docs/how_to/llm_token_usage_tracking/).generation_info.get('token_usage', {})
            self.total_tokens += usage.get('total_tokens', 0)
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)
        
        # Estimate cost (example for GPT-4o)
        self.total_cost = (
            (self.prompt_tokens / 1000 * 0.005) + 
            (self.completion_tokens / 1000 * 0.015)
        )
    
    def summary(self):
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost": self.total_cost
        }

    def invoke(self, input: str, **kwargs) -> str:
        """Synchronous invoke method for dry run"""
        if self.dry_run:
            print(f"[DRY RUN] Simulating invoke with input: {input}")
            simulated_response = f"Simulated response for: {input}"
            
            # Track invocation
            self.invocations.append({
                "input": input,
                "simulated_response": simulated_response
            })
            
            return simulated_response
        
        # If not dry run, this method should be overridden or call the original method
        raise NotImplementedError("Invoke method not implemented for non-dry run")
    
    async def ainvoke(self, input: str, **kwargs) -> str:
        """Asynchronous invoke method for dry run"""
        if self.dry_run:
            print(f"[DRY RUN] Simulating async invoke with input: {input}")
            simulated_response = f"Simulated async response for: {input}"
            
            # Track invocation
            self.invocations.append({
                "input": input,
                "simulated_response": simulated_response
            })
            
            return simulated_response
        
        # If not dry run, this method should be overridden or call the original async method
        raise NotImplementedError("Async invoke method not implemented for non-dry run")

# Usage
def run_dry_run_chain():
    # Create custom callback
    token_monitor = DryRunTokenMonitor(dry_run=True)
    
    # Configure LLM with callback
    llm = ChatOpenAI(
        model="gpt-4o",
        callbacks=[token_monitor]
    )
    
    # Simulate chain execution
    result = llm.invoke("Simulate a complex query")
    
    # Get token usage summary
    print(token_monitor.summary())

