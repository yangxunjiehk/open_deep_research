"""
Lite版本的research supervisor，专门用于开发测试
简化研究逻辑，减少迭代，提升速度
"""
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
import asyncio
from typing import Literal
from open_deep_research.configuration import Configuration
from open_deep_research.state import (
    SupervisorState,
    ResearcherState,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState
)
from open_deep_research.prompts_lite import (
    lead_researcher_lite_prompt,
    research_system_lite_prompt,
    compress_research_lite_prompt
)
from open_deep_research.utils import (
    get_today_str,
    is_token_limit_exceeded,
    get_all_tools,
    openai_websearch_called,
    anthropic_websearch_called,
    remove_up_to_last_ai_message,
    get_notes_from_tool_calls,
    get_api_key_for_model
)

# 使用相同的configurable_model
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

# 🎯 LITE版本的supervisor - 简化逻辑
async def supervisor_lite(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools_lite"]]:
    """简化的research supervisor，只进行一次研究调用"""
    print("[DEBUG] supervisor_lite - Starting with simplified logic")
    configurable = Configuration.from_runnable_config(config)
    
    # 使用轻量级模型配置以加快速度
    research_model_config = get_model_config(
        configurable.compression_model,  # 使用compression_model而不是research_model，更快
        configurable.compression_model_max_tokens, 
        config, 
        "supervisor_lite"
    )
    lead_researcher_tools = [ConductResearch, ResearchComplete]
    research_model = configurable_model.bind_tools(lead_researcher_tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    print(f"[DEBUG] supervisor_lite - Got response with {len(response.tool_calls)} tool calls")
    
    return Command(
        goto="supervisor_tools_lite",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools_lite(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_lite", "__end__"]]:
    """简化的supervisor_tools，最多只执行一轮研究"""
    print("[DEBUG] supervisor_tools_lite - Processing tool calls")
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    most_recent_message = supervisor_messages[-1]
    
    # 简化的退出条件 - 更激进的结束策略
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)
    
    # LITE模式：第一次迭代后就结束，不管结果如何
    research_iterations = state.get("research_iterations", 0)
    force_exit_lite = research_iterations >= 1  # 只允许一次迭代
    
    print(f"[DEBUG] supervisor_tools_lite - Exit conditions: no_tool_calls={no_tool_calls}, research_complete={research_complete_tool_call}, force_exit_lite={force_exit_lite}")
    
    if no_tool_calls or research_complete_tool_call or force_exit_lite:
        print("[DEBUG] supervisor_tools_lite - Exiting research")
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # 执行研究，但限制并发数为1，并使用lite研究员
    try:
        all_conduct_research_calls = [tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "ConductResearch"]
        # LITE模式：只取第一个研究调用
        conduct_research_calls = all_conduct_research_calls[:1]  # 强制限制为1个
        
        print(f"[DEBUG] supervisor_tools_lite - Executing {len(conduct_research_calls)} research calls")
        
        coros = [
            researcher_lite_subgraph.ainvoke({
                "researcher_messages": [
                    HumanMessage(content=tool_call["args"]["research_topic"])
                ],
                "research_topic": tool_call["args"]["research_topic"]
            }, config) 
            for tool_call in conduct_research_calls
        ]
        tool_results = await asyncio.gather(*coros)
        tool_messages = [ToolMessage(
                            content=observation.get("compressed_research", "Basic research completed (LITE mode)"),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        ) for observation, tool_call in zip(tool_results, conduct_research_calls)]
        
        print(f"[DEBUG] supervisor_tools_lite - Research completed, returning to supervisor")
        return Command(
            goto="supervisor_lite",
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": [f"LITE mode research completed for: {tool_call['args']['research_topic']}" for tool_call in conduct_research_calls]
            }
        )
    except Exception as e:
        print(f"[DEBUG] supervisor_tools_lite - Error occurred: {e}")
        return Command(
            goto=END,
            update={
                "notes": [f"LITE mode research encountered error: {str(e)}"],
                "research_brief": state.get("research_brief", "")
            }
        )

# 🎯 LITE版本的researcher - 简化研究
async def researcher_lite(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools_lite"]]:
    """简化的研究员，减少工具调用次数"""
    print("[DEBUG] researcher_lite - Starting simplified research")
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError("No tools found to conduct research: Please configure either your search API or add MCP tools to your configuration.")
    
    # 使用更轻量的模型配置
    research_model_config = get_model_config(
        configurable.compression_model,  # 使用compression_model而不是research_model
        configurable.compression_model_max_tokens, 
        config, 
        "researcher_lite"
    )
    researcher_system_prompt_content = research_system_lite_prompt.format(mcp_prompt=configurable.mcp_prompt or "", date=get_today_str())
    research_model = configurable_model.bind_tools(tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    response = await research_model.ainvoke([SystemMessage(content=researcher_system_prompt_content)] + researcher_messages)
    
    print(f"[DEBUG] researcher_lite - Got response with {len(response.tool_calls) if response.tool_calls else 0} tool calls")
    
    return Command(
        goto="researcher_tools_lite",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

async def execute_tool_safely(tool, args, config):
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"

async def researcher_tools_lite(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_lite", "compress_research_lite"]]:
    """简化的researcher_tools，限制工具调用次数"""
    print("[DEBUG] researcher_tools_lite - Processing tool calls")
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # 早期退出条件
    if not most_recent_message.tool_calls and not (openai_websearch_called(most_recent_message) or anthropic_websearch_called(most_recent_message)):
        print("[DEBUG] researcher_tools_lite - No tool calls, going to compress")
        return Command(goto="compress_research_lite")
    
    # LITE模式强制退出条件 - 更激进
    tool_call_iterations = state.get("tool_call_iterations", 0)
    force_exit = tool_call_iterations >= 1  # 只允许一次工具调用迭代
    
    print(f"[DEBUG] researcher_tools_lite - Iterations: {tool_call_iterations}, force_exit: {force_exit}")
    
    if force_exit:
        print("[DEBUG] researcher_tools_lite - Force exit due to LITE mode limits")
        return Command(goto="compress_research_lite")
    
    # 执行工具调用，但限制数量
    tools = await get_all_tools(config)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"):tool for tool in tools}
    tool_calls = most_recent_message.tool_calls
    
    # LITE模式：最多执行2个工具调用
    limited_tool_calls = tool_calls[:2]
    print(f"[DEBUG] researcher_tools_lite - Executing {len(limited_tool_calls)} tool calls (limited from {len(tool_calls)})")
    
    observations = []
    for tool_call in limited_tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        try:
            observation = await execute_tool_safely(tools_by_name[tool_name], tool_args, config)
            observations.append(observation)
        except Exception as e:
            print(f"[DEBUG] researcher_tools_lite - Tool error: {e}")
            observations.append(f"Error: {str(e)}")
    
    tool_outputs = [ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for observation, tool_call in zip(observations, limited_tool_calls)]
    
    # LITE模式：立即转到压缩阶段，不再迭代
    print("[DEBUG] researcher_tools_lite - Tool calls completed, going to compress")
    return Command(
        goto="compress_research_lite",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research_lite(state: ResearcherState, config: RunnableConfig):
    """简化的研究压缩，更快的处理"""
    print("[DEBUG] compress_research_lite - Compressing research findings")
    configurable = Configuration.from_runnable_config(config)
    
    # 使用更轻量的模型进行压缩
    compression_config = get_model_config(
        configurable.summarization_model,  # 使用最轻量的summarization_model
        configurable.summarization_model_max_tokens, 
        config, 
        "compress_research_lite"
    )
    synthesizer_model = configurable_model.with_config(compression_config)
    researcher_messages = state.get("researcher_messages", [])
    
    # 简化的压缩逻辑，只尝试一次
    try:
        response = await synthesizer_model.ainvoke([
            SystemMessage(content=compress_research_lite_prompt.format(date=get_today_str())),
            HumanMessage(content="Please provide a brief summary of the research findings above (LITE mode).")
        ] + researcher_messages)
        
        print("[DEBUG] compress_research_lite - Compression successful")
        return {
            "compressed_research": f"LITE MODE RESEARCH SUMMARY:\n{str(response.content)}",
            "raw_notes": [f"LITE mode research completed with {len(researcher_messages)} messages"]
        }
    except Exception as e:
        print(f"[DEBUG] compress_research_lite - Compression error: {e}")
        return {
            "compressed_research": "LITE MODE: Basic research completed, compression failed due to error",
            "raw_notes": [f"LITE mode research with error: {str(e)}"]
        }

# 构建LITE版本的子图
researcher_lite_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
researcher_lite_builder.add_node("researcher_lite", researcher_lite)
researcher_lite_builder.add_node("researcher_tools_lite", researcher_tools_lite)
researcher_lite_builder.add_node("compress_research_lite", compress_research_lite)
researcher_lite_builder.add_edge(START, "researcher_lite")
researcher_lite_builder.add_edge("compress_research_lite", END)
researcher_lite_subgraph = researcher_lite_builder.compile()

# 构建LITE版本的supervisor子图
supervisor_lite_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_lite_builder.add_node("supervisor_lite", supervisor_lite)
supervisor_lite_builder.add_node("supervisor_tools_lite", supervisor_tools_lite)
supervisor_lite_builder.add_edge(START, "supervisor_lite")
supervisor_lite_subgraph = supervisor_lite_builder.compile()

# 更新supervisor的prompt
def get_supervisor_lite_messages(research_brief: str, max_concurrent_research_units: int):
    """生成LITE版本的supervisor消息"""
    return [
        SystemMessage(content=lead_researcher_lite_prompt.format(
            date=get_today_str(),
            max_concurrent_research_units=max_concurrent_research_units
        )),
        HumanMessage(content=f"{research_brief}\n\n[LITE MODE: Please conduct ONE quick research call and finish quickly for development testing]")
    ]