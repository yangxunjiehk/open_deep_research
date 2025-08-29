"""
Deep Researcher New的LITE版本，用于快速开发测试
使用简化的research_supervisor_lite来加速研究过程
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, get_buffer_string, filter_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
import asyncio
from typing import Literal
from open_deep_research.configuration import Configuration
from open_deep_research.state import (
    AgentState,
    AgentInputState,
    SupervisorState,
    ResearcherState,
    ClarifyWithUser,
    ResearchQuestion,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    final_report_generation_prompt
)
from open_deep_research.supervisor_lite import supervisor_lite_subgraph, get_supervisor_lite_messages
from open_deep_research.utils import (
    get_today_str,
    is_token_limit_exceeded,
    get_model_token_limit,
    get_api_key_for_model
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "model_provider", "max_tokens", "api_key"),
)

def get_model_config(model_name: str, max_tokens: int, config: RunnableConfig, debug_prefix: str = ""):
    """Helper function to get model configuration for any supported provider"""
    print(f"[DEBUG] {debug_prefix} - model: {model_name}")
    api_key = get_api_key_for_model(model_name, config)
    print(f"[DEBUG] {debug_prefix} - API key obtained: {api_key[:10] if api_key else None}...")
    
    import os
    
    # Handle Azure AI Inference models (including DeepSeek on Azure)
    if model_name.startswith("azure-ai:"):
        clean_model_name = model_name.replace("azure-ai:", "")
        
        # LangChain Azure AI expects endpoint and credential via environment variables
        # AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_CREDENTIAL which we set in .env
        model_config = {
            "model": clean_model_name,
            "model_provider": "azure_ai",
            "max_tokens": max_tokens,
            "tags": ["langsmith:nostream"]
        }
        print(f"[DEBUG] {debug_prefix} - Using Azure AI Inference with model: {clean_model_name}")
        return model_config
    
    # Handle DeepSeek models (direct API)
    if model_name.startswith("deepseek:"):
        clean_model_name = model_name.replace("deepseek:", "")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_key:
            deepseek_key = api_key  # fallback to generic api_key
            
        model_config = {
            "model": clean_model_name,
            "model_provider": "deepseek",
            "max_tokens": max_tokens,
            "api_key": deepseek_key,
            "tags": ["langsmith:nostream"]
        }
        print(f"[DEBUG] {debug_prefix} - Using DeepSeek with model: {clean_model_name}")
        return model_config
    
    # Handle Azure OpenAI models
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_key and model_name.startswith("openai:"):
        # Force Azure OpenAI configuration
        clean_model_name = model_name.replace("openai:", "")
        
        # Map different models to specific Azure deployments
        deployment_mapping = {
            "gpt-5": "gpt-5-chat",
            "gpt-5-mini": "gpt-5-mini", 
            "gpt-5-nano": "gpt-5-nano",
            "gpt-4.1": "gpt-4.1",
            "gpt-4.1-mini": "gpt-4.1-mini",
            "gpt-4.1-nano": "gpt-4.1-nano",
            "gpt-4o-mini": "gpt-4o-mini"
        }
        
        deployment_name = deployment_mapping.get(clean_model_name, clean_model_name)
        
        model_config = {
            "model": deployment_name,
            "model_provider": "azure_openai", 
            "max_tokens": max_tokens,
            "api_key": azure_key,
            "tags": ["langsmith:nostream"]
        }
        print(f"[DEBUG] {debug_prefix} - Using Azure OpenAI with model: {clean_model_name}, deployment: {deployment_name}")
        return model_config
    
    # Default configuration for other providers
    model_config = {
        "model": model_name,
        "max_tokens": max_tokens,
        "api_key": api_key,
        "tags": ["langsmith:nostream"]
    }
    print(f"[DEBUG] {debug_prefix} - Using standard config with model: {model_name}")
    return model_config

# 🎯 修改：将Command模式改为直接返回字典
async def clarify_with_user_node_lite(state: AgentState, config: RunnableConfig):
    """澄清用户需求 - LITE版本，简化处理"""
    print("[DEBUG] clarify_with_user_node_lite - Starting")
    config_dict = config.get("configurable", {}) if config else {}
    allow_clarification = config_dict.get("allow_clarification", True)
    
    # LITE模式：默认跳过澄清，加速测试
    if not allow_clarification:
        print("[DEBUG] clarify_with_user_node_lite - Skipping clarification (LITE mode)")
        return {
            "clarify_status": "skipped_lite",
            "next_node": "write_research_brief"
        }
    
    messages = state["messages"]
    configurable = Configuration.from_runnable_config(config)
    
    # 使用轻量级模型
    model_config = get_model_config(
        configurable.compression_model,  # 使用compression_model而不是research_model
        configurable.compression_model_max_tokens, 
        config, 
        "clarify_with_user_lite"
    )
    model = configurable_model.with_structured_output(ClarifyWithUser).with_retry(stop_after_attempt=1).with_config(model_config)  # 减少重试次数
    
    try:
        response = await model.ainvoke([HumanMessage(content=clarify_with_user_instructions.format(messages=get_buffer_string(messages), date=get_today_str()))])
        
        if response.need_clarification:
            return {
                "messages": [AIMessage(content=response.question)],
                "clarify_status": "needs_clarification",
                "next_node": "__end__"
            }
        else:
            return {
                "messages": [AIMessage(content=response.verification)],
                "clarify_status": "completed_lite",
                "next_node": "write_research_brief"
            }
    except Exception as e:
        print(f"[DEBUG] clarify_with_user_node_lite - Error: {e}, skipping clarification")
        # LITE模式：出错时直接跳过
        return {
            "messages": [AIMessage(content="Starting research (LITE mode - clarification skipped due to error)")],
            "clarify_status": "skipped_error_lite",
            "next_node": "write_research_brief"
        }

async def write_research_brief_node_lite(state: AgentState, config: RunnableConfig):
    """编写研究纲要 - LITE版本"""
    print("[DEBUG] write_research_brief_node_lite - Starting")
    configurable = Configuration.from_runnable_config(config)
    
    # 使用轻量级模型
    research_model_config = get_model_config(
        configurable.compression_model,  # 使用compression_model而不是research_model
        configurable.compression_model_max_tokens, 
        config, 
        "write_research_brief_lite"
    )
    research_model = configurable_model.with_structured_output(ResearchQuestion).with_retry(stop_after_attempt=1).with_config(research_model_config)
    
    try:
        response = await research_model.ainvoke([HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))])
        
        # 使用LITE版本的supervisor消息
        supervisor_messages = get_supervisor_lite_messages(
            response.research_brief, 
            configurable.max_concurrent_research_units
        )
        
        print(f"[DEBUG] write_research_brief_node_lite - Research brief: {response.research_brief[:100]}...")
        
        return {
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": supervisor_messages
            },
            "brief_status": "completed_lite",
            "brief_content": response.research_brief,
            "next_node": "research_supervisor_lite"
        }
    except Exception as e:
        print(f"[DEBUG] write_research_brief_node_lite - Error: {e}")
        # LITE模式：出错时使用简单的研究纲要
        simple_brief = f"LITE MODE: Quick research on user's query (error in brief generation): {str(e)[:100]}"
        supervisor_messages = get_supervisor_lite_messages(simple_brief, 1)
        
        return {
            "research_brief": simple_brief,
            "supervisor_messages": {
                "type": "override", 
                "value": supervisor_messages
            },
            "brief_status": "error_lite",
            "brief_content": simple_brief,
            "next_node": "research_supervisor_lite"
        }

# 🎯 修改：将final_report_generation改为直接返回字典，使用轻量级处理
async def final_report_generation_node_lite(state: AgentState, config: RunnableConfig):
    """生成最终报告 - LITE版本，简化报告生成"""
    print("[DEBUG] final_report_generation_node_lite - Starting")
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []},}
    configurable = Configuration.from_runnable_config(config)
    
    # 使用轻量级模型
    writer_model_config = get_model_config(
        configurable.summarization_model,  # 使用最轻量的模型
        configurable.summarization_model_max_tokens,  # 减少token数量 
        config, 
        "final_report_generation_lite"
    )
    
    findings = "\n".join(notes) if notes else "No detailed findings available (LITE mode)"
    
    # 简化的final report prompt
    simple_final_report_prompt = f"""Based on the research brief and findings below, create a concise summary report (LITE MODE for development):

Research Brief: {state.get('research_brief', 'Not available')}

User Messages: {get_buffer_string(state.get('messages', []))}

Research Findings: {findings}

Please create a brief, structured answer that:
1. Addresses the main question/topic
2. Includes key findings from the research  
3. Is concise but informative (this is LITE mode for development testing)

Today's date is {get_today_str()}."""

    try:
        final_report = await configurable_model.with_config(writer_model_config).ainvoke([HumanMessage(content=simple_final_report_prompt)])
        
        report_content = str(final_report.content)
        lite_report = f"# LITE MODE RESEARCH REPORT\n\n{report_content}\n\n---\n*This is a LITE mode report generated for development testing. For comprehensive research, use the full version.*"
        
        print("[DEBUG] final_report_generation_node_lite - Report generated successfully")
        
        return {
            "final_report": lite_report, 
            "messages": [AIMessage(content=lite_report)],
            "report_status": "completed_lite",
            "report_content": lite_report,
            **cleared_state
        }
    except Exception as e:
        print(f"[DEBUG] final_report_generation_node_lite - Error: {e}")
        error_report = f"# LITE MODE RESEARCH REPORT - ERROR\n\nAn error occurred while generating the report: {str(e)}\n\nResearch Brief: {state.get('research_brief', 'Not available')}\n\nBasic Findings: {findings[:500] if findings else 'No findings available'}\n\n---\n*This is an error report from LITE mode. Please check the configuration and try again.*"
        
        return {
            "final_report": error_report,
            "messages": [AIMessage(content=error_report)],
            "report_status": "error_lite",
            "report_content": error_report,
            **cleared_state
        }

# 🎯 添加条件边函数
def should_continue_after_clarify_lite(state: AgentState) -> str:
    """决定澄清后的路径"""
    next_node = state.get("next_node", "write_research_brief")
    return next_node

def should_continue_after_brief_lite(state: AgentState) -> str:
    """决定研究纲要后的路径"""
    next_node = state.get("next_node", "research_supervisor_lite")
    return next_node

# 构建新的LITE graph
deep_researcher_new_lite_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
deep_researcher_new_lite_builder.add_node("clarify_with_user", clarify_with_user_node_lite)
deep_researcher_new_lite_builder.add_node("write_research_brief", write_research_brief_node_lite)
deep_researcher_new_lite_builder.add_node("research_supervisor_lite", supervisor_lite_subgraph)
deep_researcher_new_lite_builder.add_node("final_report_generation", final_report_generation_node_lite)

# 添加条件边
deep_researcher_new_lite_builder.add_edge(START, "clarify_with_user")
deep_researcher_new_lite_builder.add_conditional_edges("clarify_with_user", should_continue_after_clarify_lite)
deep_researcher_new_lite_builder.add_conditional_edges("write_research_brief", should_continue_after_brief_lite)
deep_researcher_new_lite_builder.add_edge("research_supervisor_lite", "final_report_generation")
deep_researcher_new_lite_builder.add_edge("final_report_generation", END)

deep_researcher_new_lite = deep_researcher_new_lite_builder.compile()