from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, get_buffer_string, filter_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
import asyncio
from typing import Literal
from open_deep_research.configuration import (
    Configuration, 
)
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
    research_system_prompt,
    compress_research_system_prompt,
    compress_research_simple_human_message,
    final_report_generation_prompt,
    lead_researcher_prompt
)
from open_deep_research.utils import (
    get_today_str,
    is_token_limit_exceeded,
    get_model_token_limit,
    get_all_tools,
    openai_websearch_called,
    anthropic_websearch_called,
    remove_up_to_last_ai_message,
    get_api_key_for_model,
    get_notes_from_tool_calls
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
async def clarify_with_user_node(state: AgentState, config: RunnableConfig):
    """澄清用户需求 - 修改为支持前端进度显示"""
    config_dict = config.get("configurable", {}) if config else {}
    allow_clarification = config_dict.get("allow_clarification", True)
    
    if not allow_clarification:
        return {
            "clarify_status": "skipped",
            "next_node": "write_research_brief"
        }
    
    messages = state["messages"]
    configurable = Configuration.from_runnable_config(config)
    
    model_config = get_model_config(
        configurable.research_model, 
        configurable.research_model_max_tokens, 
        config, 
        "clarify_with_user"
    )
    model = configurable_model.with_structured_output(ClarifyWithUser).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(model_config)
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
            "clarify_status": "completed",
            "next_node": "write_research_brief"
        }

async def write_research_brief_node(state: AgentState, config: RunnableConfig):
    """编写研究纲要 - 修改为支持前端进度显示"""
    configurable = Configuration.from_runnable_config(config)
    
    research_model_config = get_model_config(
        configurable.research_model, 
        configurable.research_model_max_tokens, 
        config, 
        "write_research_brief"
    )
    research_model = configurable_model.with_structured_output(ResearchQuestion).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    response = await research_model.ainvoke([HumanMessage(content=transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    ))])
    
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": {
            "type": "override",
            "value": [
                SystemMessage(content=lead_researcher_prompt.format(
                    date=get_today_str(),
                    max_concurrent_research_units=configurable.max_concurrent_research_units
                )),
                HumanMessage(content=response.research_brief)
            ]
        },
        "brief_status": "completed",
        "brief_content": response.research_brief,
        "next_node": "research_supervisor"
    }

# 原始的supervisor和相关函数保持不变，因为它们是子图
async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    
    research_model_config = get_model_config(
        configurable.research_model, 
        configurable.research_model_max_tokens, 
        config, 
        "supervisor"
    )
    lead_researcher_tools = [ConductResearch, ResearchComplete]
    research_model = configurable_model.bind_tools(lead_researcher_tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    # Exit Criteria
    # 1. We have exceeded our max guardrail research iterations
    # 2. No tool calls were made by the supervisor
    # 3. The most recent message contains a ResearchComplete tool call and there is only one tool call in the message
    exceeded_allowed_iterations = research_iterations >= configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    # Otherwise, conduct research and gather results.
    try:
        all_conduct_research_calls = [tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "ConductResearch"]
        conduct_research_calls = all_conduct_research_calls[:configurable.max_concurrent_research_units]
        overflow_conduct_research_calls = all_conduct_research_calls[configurable.max_concurrent_research_units:]
        coros = [
            researcher_subgraph.ainvoke({
                "researcher_messages": [
                    HumanMessage(content=tool_call["args"]["research_topic"])
                ],
                "research_topic": tool_call["args"]["research_topic"]
            }, config) 
            for tool_call in conduct_research_calls
        ]
        tool_results = await asyncio.gather(*coros)
        tool_messages = [ToolMessage(
                            content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        ) for observation, tool_call in zip(tool_results, conduct_research_calls)]
        # Handle any tool calls made > max_concurrent_research_units
        for overflow_conduct_research_call in overflow_conduct_research_calls:
            tool_messages.append(ToolMessage(
                content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                name="ConductResearch",
                tool_call_id=overflow_conduct_research_call["id"]
            ))
        raw_notes_concat = "\n".join(["\n".join(observation.get("raw_notes", [])) for observation in tool_results])
        return Command(
            goto="supervisor",
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": [raw_notes_concat]
            }
        )
    except Exception as e:
        if is_token_limit_exceeded(e, configurable.research_model):
            print(f"Token limit exceeded while reflecting: {e}")
        else:
            print(f"Other error in reflection phase: {e}")
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )


supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError("No tools found to conduct research: Please configure either your search API or add MCP tools to your configuration.")
    
    research_model_config = get_model_config(
        configurable.research_model, 
        configurable.research_model_max_tokens, 
        config, 
        "researcher"
    )
    researcher_system_prompt = research_system_prompt.format(mcp_prompt=configurable.mcp_prompt or "", date=get_today_str())
    research_model = configurable_model.bind_tools(tools).with_retry(stop_after_attempt=configurable.max_structured_output_retries).with_config(research_model_config)
    response = await research_model.ainvoke([SystemMessage(content=researcher_system_prompt)] + researcher_messages)
    return Command(
        goto="researcher_tools",
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


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    # Early Exit Criteria: No tool calls (or native web search calls)were made by the researcher
    if not most_recent_message.tool_calls and not (openai_websearch_called(most_recent_message) or anthropic_websearch_called(most_recent_message)):
        return Command(
            goto="compress_research",
        )
    # Otherwise, execute tools and gather results.
    tools = await get_all_tools(config)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"):tool for tool in tools}
    tool_calls = most_recent_message.tool_calls
    
    # 执行工具调用，并在执行过程中记录详细信息
    observations = []
    tool_execution_details = []
    
    print(f"[DEBUG] Total tool calls to execute: {len(tool_calls)}")
    for tool_call in tool_calls:
        print(f"[DEBUG] Tool call: {tool_call}")
    
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        
        print(f"[DEBUG] Starting tool execution: {tool_name} ({i + 1}/{len(tool_calls)}) with args: {tool_args}")
        
        # 立即发送工具开始执行的实时事件 - 通过状态更新机制
        starting_detail = {
            "tool_name": tool_name,
            "tool_index": i + 1,
            "total_tools": len(tool_calls),
            "status": "starting",
            "args": tool_args,
            "timestamp": f"Tool {i + 1}/{len(tool_calls)} starting"
        }
        tool_execution_details.append(starting_detail)
        
        print(f"[DEBUG] Recording starting status for {tool_name}")
        
        try:
            print(f"[DEBUG] About to call execute_tool_safely for {tool_name}")
            # 执行工具
            observation = await execute_tool_safely(tools_by_name[tool_name], tool_args, config)
            print(f"[DEBUG] execute_tool_safely returned for {tool_name}")
            observations.append(observation)
            
            # 工具完成后的详情
            completed_detail = {
                "tool_name": tool_name,
                "tool_index": i + 1,
                "total_tools": len(tool_calls),
                "status": "completed",
                "args": tool_args,
                "result_length": len(str(observation)) if observation else 0,
                "timestamp": f"Tool {i + 1}/{len(tool_calls)} completed"
            }
            tool_execution_details.append(completed_detail)
            
            
            print(f"[DEBUG] Completed tool execution: {tool_name}, result length: {completed_detail['result_length']}")
            
        except Exception as e:
            print(f"[DEBUG] Error in tool execution: {tool_name}, error: {e}")
            import traceback
            print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            # 即使出错也记录到详情中
            error_detail = {
                "tool_name": tool_name,
                "tool_index": i + 1,
                "total_tools": len(tool_calls),
                "status": "error",
                "args": tool_args,
                "error": str(e),
                "timestamp": f"Tool {i + 1}/{len(tool_calls)} error"
            }
            tool_execution_details.append(error_detail)
            observations.append(f"Error: {str(e)}")
    
    tool_outputs = [ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for observation, tool_call in zip(observations, tool_calls)]
    
    print(f"[DEBUG] Prepared tool_execution_details with {len(tool_execution_details)} items:")
    for detail in tool_execution_details:
        print(f"[DEBUG] Tool detail: {detail}")
    
    # Late Exit Criteria: We have exceeded our max guardrail tool call iterations or the most recent message contains a ResearchComplete tool call
    # These are late exit criteria because we need to add ToolMessages
    if state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls or any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls):
        print(f"[DEBUG] Returning final Command to compress_research with tool_execution_details")
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
                "tool_execution_details": tool_execution_details,  # 添加工具执行详情
                "current_tool_status": "所有工具执行完成，正在压缩研究结果..."
            }
        )
    else:
        print(f"[DEBUG] Returning final Command to researcher with tool_execution_details")
        return Command(
            goto="researcher",
            update={
                "researcher_messages": tool_outputs,
                "tool_execution_details": tool_execution_details,  # 添加工具执行详情
                "current_tool_status": "所有工具执行完成，继续研究..."
            }
        )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    synthesis_attempts = 0
    compression_config = get_model_config(
        configurable.compression_model, 
        configurable.compression_model_max_tokens, 
        config, 
        "compress_research"
    )
    synthesizer_model = configurable_model.with_config(compression_config)
    researcher_messages = state.get("researcher_messages", [])
    # Update the system prompt to now focus on compression rather than research.
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    while synthesis_attempts < 3:
        try:
            response = await synthesizer_model.ainvoke([SystemMessage(content=compress_research_system_prompt.format(date=get_today_str()))] + researcher_messages)
            return {
                "compressed_research": str(response.content),
                "raw_notes": ["\n".join([str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])])]
            }
        except Exception as e:
            synthesis_attempts += 1
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                print(f"Token limit exceeded while synthesizing: {e}. Pruning the messages to try again.")
                continue         
            print(f"Error synthesizing research report: {e}")
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": ["\n".join([str(m.content) for m in filter_messages(researcher_messages, include_types=["tool", "ai"])])]
    }


researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()

# 🎯 修改：将final_report_generation改为直接返回字典
async def final_report_generation_node(state: AgentState, config: RunnableConfig):
    """生成最终报告 - 修改为支持前端进度显示"""
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []},}
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = get_model_config(
        configurable.final_report_model, 
        configurable.final_report_model_max_tokens, 
        config, 
        "final_report_generation"
    )
    
    findings = "\n".join(notes)
    max_retries = 3
    current_retry = 0
    while current_retry <= max_retries:
        final_report_prompt = final_report_generation_prompt.format(
            research_brief=state.get("research_brief", ""),
            messages=get_buffer_string(state.get("messages", [])),
            findings=findings,
            date=get_today_str()
        )
        try:
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([HumanMessage(content=final_report_prompt)])
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                "report_status": "completed",
                "report_content": final_report.content,
                **cleared_state
            }
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                if current_retry == 0:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "report_status": "error",
                            **cleared_state
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)
                print("Reducing the chars to", findings_token_limit)
                findings = findings[:findings_token_limit]
                current_retry += 1
            else:
                # If not a token limit exceeded error, then we just throw an error.
                return {
                    "final_report": f"Error generating final report: {e}",
                    "report_status": "error",
                    **cleared_state
                }
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [final_report],
        "report_status": "error",
        **cleared_state
    }

# 🎯 添加条件边函数
def should_continue_after_clarify(state: AgentState) -> str:
    """决定澄清后的路径"""
    next_node = state.get("next_node", "write_research_brief")
    return next_node

def should_continue_after_brief(state: AgentState) -> str:
    """决定研究纲要后的路径"""
    next_node = state.get("next_node", "research_supervisor")
    return next_node

# 构建新的graph
deep_researcher_new_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
deep_researcher_new_builder.add_node("clarify_with_user", clarify_with_user_node)
deep_researcher_new_builder.add_node("write_research_brief", write_research_brief_node)
deep_researcher_new_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_new_builder.add_node("final_report_generation", final_report_generation_node)

# 添加条件边
deep_researcher_new_builder.add_edge(START, "clarify_with_user")
deep_researcher_new_builder.add_conditional_edges("clarify_with_user", should_continue_after_clarify)
deep_researcher_new_builder.add_conditional_edges("write_research_brief", should_continue_after_brief)
deep_researcher_new_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_new_builder.add_edge("final_report_generation", END)

deep_researcher_new = deep_researcher_new_builder.compile()