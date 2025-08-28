"""
Flat Graph Implementation - 单层图架构
将原来的多层嵌套图结构扁平化，让所有节点都在主图级别，便于前端接收所有节点的事件
"""

import os
from typing import Literal, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import Command
from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic

from open_deep_research.configuration import Configuration
from open_deep_research.state import (
    AgentState,
    AgentInputState,
    ResearchQuestion,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    research_system_prompt,
    compress_research_system_prompt,
    compress_research_simple_human_message,
    final_report_generation_prompt,
    lead_researcher_prompt,
)
from open_deep_research.utils import (
    get_today_str,
    is_token_limit_exceeded,
    get_notes_from_tool_calls,
    get_all_tools,
    openai_websearch_called,
    anthropic_websearch_called,
)

def update_step_status(step_name: str, status: str):
    """辅助函数：创建步骤状态更新"""
    return {
        "step_status": {"type": "override", "value": {step_name: status}},
        "current_step": step_name
    }


def get_model_config(model_name: str, max_tokens: int, config: RunnableConfig, debug_prefix: str = "") -> dict:
    """获取模型配置"""
    model_config = {
        "configurable": config.get("configurable", {}),
        "recursion_limit": config.get("recursion_limit", 20),
        "tags": ["langsmith:nostream"]
    }
    print(f"[DEBUG] {debug_prefix} - Using model: {model_name}")
    return model_config


def get_azure_openai_llm(model_name: str, temperature: float = 0.3):
    """创建Azure OpenAI模型实例"""
    import os
    
    # 从环境变量获取Azure配置
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    
    if not azure_endpoint or not azure_api_key:
        raise ValueError("Azure OpenAI configuration missing. Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
    
    return AzureChatOpenAI(
        azure_deployment=model_name,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_version,
        temperature=temperature,
        max_retries=2,
    )

def get_configurable_model(model_name: str, temperature: float = 0.3, config=None):
    """根据模型名称获取相应的模型实例"""
    from open_deep_research.utils import get_api_key_for_model
    
    # 对于Azure OpenAI模型
    if model_name.startswith("openai:"):
        actual_model_name = model_name.split(":", 1)[1]
        return get_azure_openai_llm(actual_model_name, temperature)
    # 对于Anthropic模型
    elif model_name.startswith("anthropic:"):
        api_key = get_api_key_for_model(model_name, config or {})
        return ChatAnthropic(
            model=model_name.split(":", 1)[1],
            api_key=api_key,
            temperature=temperature,
            max_retries=2,
        )
    else:
        # 默认使用Azure OpenAI
        return get_azure_openai_llm(model_name, temperature)


# ===== 主图节点 - 扁平化所有重要节点 =====

async def clarify_with_user_node(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """澄清用户需求"""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")
    
    messages = state["messages"]
    
    model_config = get_model_config(
        configurable.research_model, 
        configurable.research_model_max_tokens, 
        config, 
        "clarify_with_user"
    )
    
    configurable_model = get_configurable_model(configurable.research_model, config=config)
    model = configurable_model.with_structured_output(ClarifyWithUser).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    ).with_config(model_config)
    
    response = await model.ainvoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages), 
            date=get_today_str()
        ))
    ])
    
    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})


async def write_research_brief_node(state: AgentState, config: RunnableConfig) -> Command[Literal["plan_research"]]:
    """编写研究简报"""
    configurable = Configuration.from_runnable_config(config)
    
    research_model_config = get_model_config(
        configurable.research_model, 
        configurable.research_model_max_tokens, 
        config, 
        "write_research_brief"
    )
    
    configurable_model = get_configurable_model(configurable.research_model, config=config)
    research_model = configurable_model.with_structured_output(ResearchQuestion).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    ).with_config(research_model_config)
    
    response = await research_model.ainvoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])
    
    return Command(
        goto="plan_research",
        update={
            "messages": [AIMessage(content=response.research_brief)],
            "research_brief": response.research_brief,
        }
    )


async def plan_research_node(state: AgentState, config: RunnableConfig) -> Command[Literal["execute_research_tools"]]:
    """规划研究 - 相当于原来的supervisor节点"""
    configurable = Configuration.from_runnable_config(config)
    
    model_config = get_model_config(
        configurable.research_model, 
        configurable.research_model_max_tokens, 
        config, 
        "plan_research"
    )
    
    lead_researcher_tools = [ConductResearch, ResearchComplete]
    configurable_model = get_configurable_model(configurable.research_model, config=config)
    research_model = configurable_model.bind_tools(lead_researcher_tools).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    ).with_config(model_config)
    
    # 获取和处理supervisor消息历史
    supervisor_messages = state.get("supervisor_messages", [])
    if not supervisor_messages:
        # 首次调用，初始化消息
        research_brief = state.get("research_brief", "")
        supervisor_messages = [
            SystemMessage(content=lead_researcher_prompt.format(
                date=get_today_str(),
                max_concurrent_research_units=configurable.max_concurrent_research_units
            )),
            HumanMessage(content=research_brief)
        ]
    
    response = await research_model.ainvoke(supervisor_messages)
    
    # 使用 override 模式设置完整的消息历史
    updated_supervisor_messages = supervisor_messages + [response]
    
    return Command(
        goto="execute_research_tools",
        update={
            "supervisor_messages": {"type": "override", "value": updated_supervisor_messages},
        }
    )


async def execute_research_tools_node(state: AgentState, config: RunnableConfig) -> Command[Literal["perform_searches", "compress_research", "__end__"]]:
    """执行研究工具 - 相当于原来的supervisor_tools节点"""
    from langgraph.config import get_stream_writer
    import time
    
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    most_recent_message = supervisor_messages[-1]
    
    # 早期退出：没有工具调用
    if not most_recent_message.tool_calls:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # 检查是否有ResearchComplete调用
    if any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls):
        return Command(
            goto="compress_research",
            update={}
        )
    
    # 🚀 发送搜索开始状态到前端
    try:
        writer = get_stream_writer()
        if writer:
            writer({
                "event_type": "step_status",
                "step": "web_search",
                "status": "started",
                "message": "准备开始 Tavily 搜索...",
                "timestamp": time.time(),
                "tool_calls_count": len(most_recent_message.tool_calls)
            })
    except Exception as e:
        print(f"Failed to send stream event: {e}")
    
    # 准备进行研究
    return Command(
        goto="perform_searches",
        update={
            "tool_calls": most_recent_message.tool_calls,
            "current_research_step": 0
        }
    )


async def perform_searches_node(state: AgentState, config: RunnableConfig) -> Command[Literal["analyze_search_results", "plan_research"]]:
    """执行搜索 - 这是最重要的节点，执行Tavily搜索"""
    from langgraph.config import get_stream_writer
    import time
    
    configurable = Configuration.from_runnable_config(config)
    
    # 获取要执行的工具调用
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return Command(goto="plan_research")
    
    # 🚀 发送搜索开始进行状态
    try:
        writer = get_stream_writer()
        if writer:
            writer({
                "event_type": "step_status",
                "step": "web_search",
                "status": "in_progress",
                "message": f"正在执行 {len(tool_calls)} 个 Tavily 搜索...",
                "timestamp": time.time()
            })
    except Exception as e:
        print(f"Failed to send stream event: {e}")
    
    # 执行Tavily搜索
    tools = await get_all_tools(config)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool for tool in tools}
    
    
    search_results = []
    for i, tool_call in enumerate(tool_calls):
        if tool_call["name"] == "ConductResearch":
            research_topic = tool_call["args"].get("research_topic", "")
            
            # 截断查询以符合 Tavily 的 400 字符限制
            if len(research_topic) > 400:
                research_topic = research_topic[:400]
            
            # 🚀 发送单个搜索进度
            try:
                if writer:
                    writer({
                        "event_type": "search_progress",
                        "current": i + 1,
                        "total": len(tool_calls),
                        "query": research_topic[:50] + "..." if len(research_topic) > 50 else research_topic,
                        "message": f"搜索 {i + 1}/{len(tool_calls)}: {research_topic[:50]}...",
                        "timestamp": time.time()
                    })
            except Exception as e:
                print(f"Failed to send progress event: {e}")
            
            # 这里执行实际的搜索
            # 为了简化，我们直接调用tavily_search工具
            if "tavily_search" in tools_by_name:
                try:
                    result = await tools_by_name["tavily_search"].ainvoke(
                        {"queries": [research_topic]}, 
                        config
                    )
                    search_results.append({
                        "topic": research_topic,
                        "result": result,
                        "tool_call_id": tool_call["id"]
                    })
                except Exception as e:
                    search_results.append({
                        "topic": research_topic,
                        "error": str(e),
                        "tool_call_id": tool_call["id"]
                    })
    
    # 🚀 发送搜索完成和分析开始状态
    try:
        if writer:
            writer({
                "event_type": "step_status",
                "step": "web_search",
                "status": "completed",
                "message": f"Tavily 搜索完成，获得 {len(search_results)} 个结果",
                "timestamp": time.time(),
                "results_count": len(search_results)
            })
            
            writer({
                "event_type": "step_status", 
                "step": "search_analysis",
                "status": "started",
                "message": "开始分析搜索结果...",
                "timestamp": time.time()
            })
    except Exception as e:
        print(f"Failed to send completion event: {e}")
    
    return Command(
        goto="analyze_search_results",
        update={
            "search_results": search_results,
            "raw_notes": [str(r.get("result", "")) for r in search_results]
        }
    )


async def analyze_search_results_node(state: AgentState, config: RunnableConfig) -> Command[Literal["compress_research", "plan_research"]]:
    """分析搜索结果 - 总结和整理搜索结果"""
    from langgraph.config import get_stream_writer
    import time
    
    configurable = Configuration.from_runnable_config(config)
    search_results = state.get("search_results", [])
    
    # 🚀 发送分析进行中状态
    try:
        writer = get_stream_writer()
        if writer:
            writer({
                "event_type": "step_status",
                "step": "search_analysis", 
                "status": "in_progress",
                "message": f"正在分析 {len(search_results)} 个搜索结果...",
                "timestamp": time.time()
            })
    except Exception as e:
        print(f"Failed to send analysis progress event: {e}")
    
    if not search_results:
        return Command(goto="plan_research")
    
    # 构建工具消息反馈给supervisor
    tool_messages = []
    for result in search_results:
        if "error" in result:
            content = f"Error researching {result['topic']}: {result['error']}"
        else:
            # 直接使用搜索结果（不进行额外总结以简化）
            content = f"Research on {result['topic']}:\n{str(result.get('result', ''))[:2000]}"
        
        tool_messages.append(ToolMessage(
            content=content,
            name="ConductResearch",
            tool_call_id=result["tool_call_id"]
        ))
    
    # 🚀 发送分析完成状态
    try:
        if writer:
            writer({
                "event_type": "step_status",
                "step": "search_analysis",
                "status": "completed", 
                "message": f"搜索结果分析完成，处理了 {len(search_results)} 个结果",
                "timestamp": time.time()
            })
    except Exception as e:
        print(f"Failed to send analysis completion event: {e}")
    
    # 检查是否需要继续研究
    iterations = state.get("research_iterations", 0) + 1
    if iterations >= configurable.max_researcher_iterations:
        return Command(
            goto="compress_research",
            update={
                "supervisor_messages": tool_messages,
                "research_iterations": iterations
            }
        )
    
    # 使用 override 模式设置完整的消息历史，避免重复累积
    current_supervisor_messages = state.get("supervisor_messages", [])
    updated_supervisor_messages = current_supervisor_messages + tool_messages
    
    return Command(
        goto="plan_research",
        update={
            "supervisor_messages": {"type": "override", "value": updated_supervisor_messages},
            "research_iterations": iterations
        }
    )


async def compress_research_node(state: AgentState, config: RunnableConfig) -> Command[Literal["generate_final_report"]]:
    """压缩研究结果"""
    configurable = Configuration.from_runnable_config(config)
    notes = state.get("raw_notes", [])
    
    if not notes:
        notes = get_notes_from_tool_calls(state.get("supervisor_messages", []))
    
    compression_config = get_model_config(
        configurable.compression_model, 
        configurable.compression_model_max_tokens, 
        config, 
        "compress_research"
    )
    
    configurable_model = get_configurable_model(configurable.compression_model, config=config)
    compression_model = configurable_model.with_config(compression_config)
    
    compressed_notes = []
    for note in notes:
        if note:
            try:
                response = await compression_model.ainvoke([
                    SystemMessage(content=compress_research_system_prompt),
                    HumanMessage(content=f"{note}\n\n{compress_research_simple_human_message}")
                ])
                compressed_notes.append(response.content)
            except Exception as e:
                print(f"[DEBUG] Compression error: {e}")
                compressed_notes.append(note[:2000])  # 截断以防太长
    
    return Command(
        goto="generate_final_report",
        update={
            "notes": compressed_notes
        }
    )


async def generate_final_report_node(state: AgentState, config: RunnableConfig):
    """生成最终报告"""
    notes = state.get("notes", [])
    configurable = Configuration.from_runnable_config(config)
    
    writer_model_config = get_model_config(
        configurable.final_report_model, 
        configurable.final_report_model_max_tokens, 
        config, 
        "generate_final_report"
    )
    
    findings = "\n".join(notes)
    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        messages=get_buffer_string(state.get("messages", [])),
        findings=findings,
        date=get_today_str()
    )
    
    configurable_model = get_configurable_model(configurable.final_report_model, config=config)
    writer_model = configurable_model.with_config(writer_model_config)
    
    final_report = await writer_model.ainvoke([
        HumanMessage(content=final_report_prompt)
    ])
    
    # 清理状态
    cleared_state = {
        "notes": {"type": "override", "value": []},
        "raw_notes": {"type": "override", "value": []},
        "search_results": {"type": "override", "value": []},
        "supervisor_messages": {"type": "override", "value": []},
    }
    
    return {
        "messages": [final_report],
        **cleared_state
    }


# ===== 构建单层图 =====
def build_flat_graph():
    """构建扁平化的单层图"""
    builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
    
    # 添加所有节点到主图级别
    builder.add_node("clarify_with_user", clarify_with_user_node)
    builder.add_node("write_research_brief", write_research_brief_node)
    builder.add_node("plan_research", plan_research_node)
    builder.add_node("execute_research_tools", execute_research_tools_node)
    builder.add_node("perform_searches", perform_searches_node)  # 关键节点：执行Tavily搜索
    builder.add_node("analyze_search_results", analyze_search_results_node)
    builder.add_node("compress_research", compress_research_node)
    builder.add_node("generate_final_report", generate_final_report_node)
    
    # 添加边
    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("generate_final_report", END)
    
    return builder.compile()


# 导出扁平图实例
flat_researcher = build_flat_graph()