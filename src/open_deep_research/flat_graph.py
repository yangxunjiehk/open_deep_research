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

async def clarify_with_user_node(state: AgentState, config: RunnableConfig):
    """澄清用户需求"""
    # 直接检查传入的configurable参数，优先级高于默认配置
    config_dict = config.get("configurable", {}) if config else {}
    allow_clarification = config_dict.get("allow_clarification", True)
    
    if not allow_clarification:
        print("[DEBUG] Skipping clarification as requested by frontend")
        return {
            "clarify_status": "skipped",
            "should_continue": True
        }
    
    configurable = Configuration.from_runnable_config(config)
    
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
        return {
            "messages": [AIMessage(content=response.question)],
            "clarify_status": "needed",
            "should_continue": False
        }
    else:
        return {
            "messages": [AIMessage(content=response.verification)],
            "clarify_status": "completed", 
            "should_continue": True
        }


async def write_research_brief_node(state: AgentState, config: RunnableConfig):
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
    
    return {
        "messages": [AIMessage(content=response.research_brief)],
        "research_brief": response.research_brief,
        "research_brief_status": "completed"
    }


async def generate_queries_node(state: AgentState, config: RunnableConfig):
    """生成搜索查询 - 匹配前端的generate_query事件"""
    research_brief = state.get("research_brief", "")
    # 根据研究简报生成相关查询
    sample_queries = [
        f"latest developments in {research_brief[:50]}",
        f"market analysis {research_brief[:50]}",
        f"industry trends {research_brief[:50]}"
    ]
    
    # 直接返回状态更新，不使用Command模式
    return {
        "query_list": sample_queries,
        "generate_query_status": "generated"
    }


async def plan_research_node(state: AgentState, config: RunnableConfig):
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
    
    # 构建计划数据
    tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
    plan_data = [{"description": f"研究任务: {tc.get('args', {}).get('research_topic', 'Unknown')}"} 
                 for tc in tool_calls if tc.get('name') == 'ConductResearch']
    
    return {
        "supervisor_messages": updated_supervisor_messages,
        "planner_node": {
            "plan": plan_data if plan_data else [{"description": "正在分析研究需求并制定研究计划..."}]
        },
        "planner": {
            "plan": plan_data
        },
        "tool_calls": tool_calls,
        "planning_status": "completed"
    }


async def execute_research_tools_node(state: AgentState, config: RunnableConfig):
    """执行研究工具 - 相当于原来的supervisor_tools节点"""
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    
    if not supervisor_messages:
        return {
            "execution_status": "no_messages",
            "should_end": True
        }
    
    most_recent_message = supervisor_messages[-1]
    
    # 早期退出：没有工具调用
    if not most_recent_message.tool_calls:
        return {
            "notes": get_notes_from_tool_calls(supervisor_messages),
            "research_brief": state.get("research_brief", ""),
            "execution_status": "no_tool_calls",
            "should_end": True
        }
    
    # 检查是否有ResearchComplete调用
    has_research_complete = any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)
    
    if has_research_complete:
        return {
            "execution_status": "research_complete",
            "should_compress": True
        }
    
    # 构建任务记录数据
    research_tasks = []
    for tool_call in most_recent_message.tool_calls:
        if tool_call["name"] == "ConductResearch":
            topic = tool_call["args"].get("research_topic", "")
            research_tasks.append({
                "description": topic,
                "status": "starting"
            })
    
    # 准备进行研究
    return {
        "tool_calls": most_recent_message.tool_calls,
        "current_research_step": 0,
        "record_task_completion": {
            "ledger": research_tasks,
            "next_node_decision": "continue",
            "status": "准备执行研究任务"
        },
        "execution_status": "ready_for_search",
        "should_search": True
    }


async def perform_searches_node(state: AgentState, config: RunnableConfig):
    """执行搜索 - 这是最重要的节点，执行Tavily搜索"""
    configurable = Configuration.from_runnable_config(config)
    
    # 获取要执行的工具调用
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return Command(goto="plan_research")
    
    # 执行Tavily搜索
    tools = await get_all_tools(config)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool for tool in tools}
    
    search_results = []
    sources_gathered = []
    for i, tool_call in enumerate(tool_calls):
        if tool_call["name"] == "ConductResearch":
            research_topic = tool_call["args"].get("research_topic", "")
            
            # 截断查询以符合 Tavily 的 400 字符限制
            if len(research_topic) > 400:
                research_topic = research_topic[:400]
            
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
                    
                    # 构造sources_gathered格式，匹配前端期望
                    if isinstance(result, str) and result:
                        sources_gathered.append({
                            "title": f"Research Result for: {research_topic[:30]}",
                            "url": "https://tavily.com/search",
                            "snippet": result[:200] + "..." if len(result) > 200 else result,
                            "label": research_topic[:30]
                        })
                        
                except Exception as e:
                    search_results.append({
                        "topic": research_topic,
                        "error": str(e),
                        "tool_call_id": tool_call["id"]
                    })
    
    return {
        "search_results": search_results,
        "raw_notes": [str(r.get("result", "")) for r in search_results],
        "sources_gathered": sources_gathered,
        "web_research_status": "completed",
        "web_research_results_count": len(search_results)
    }


async def analyze_search_results_node(state: AgentState, config: RunnableConfig):
    """分析搜索结果 - 总结和整理搜索结果"""
    configurable = Configuration.from_runnable_config(config)
    search_results = state.get("search_results", [])
    
    # 评估研究是否充足
    is_sufficient = len(search_results) >= 2  # 简单的充足性判断
    
    if not search_results:
        return {
            "analysis_status": "no_results",
            "should_continue_research": True,
            "reflection_is_sufficient": False,
            "reflection_follow_up_queries": ["需要更多研究数据"]
        }
    
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
    
    # 检查是否需要继续研究
    iterations = state.get("research_iterations", 0) + 1
    final_is_sufficient = is_sufficient and iterations >= configurable.max_researcher_iterations
    
    return {
        "supervisor_messages": state.get("supervisor_messages", []) + tool_messages,
        "research_iterations": iterations,
        "reflection_is_sufficient": final_is_sufficient,
        "reflection_follow_up_queries": [] if final_is_sufficient else [f"继续深入研究第{iterations}轮"],
        "analysis_status": "completed" if final_is_sufficient else "need_more_research",
        "should_compress": final_is_sufficient,
        "should_continue_research": not final_is_sufficient
    }


def should_continue_research(state: AgentState) -> str:
    """决定是否继续研究还是结束"""
    iterations = state.get("research_iterations", 0)
    configurable = Configuration()  # 使用默认配置
    
    if iterations >= configurable.max_researcher_iterations:
        return "compress_research"
    else:
        return "plan_research"


async def compress_research_node(state: AgentState, config: RunnableConfig):
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
    
    return {
        "notes": compressed_notes,
        "compression_status": "completed",
        "compressed_notes_count": len(compressed_notes)
    }


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
    
    return {
        "messages": [final_report],
        "notes": [],
        "raw_notes": [],
        "search_results": [],
        "supervisor_messages": [],
        "finalize_answer": {
            "status": "completed",
            "message": "研究报告生成完成"
        },
        "final_report_status": "completed"
    }


# ===== 条件边函数 =====
def should_continue_after_clarify(state: AgentState) -> str:
    """决定澄清后的路径"""
    if state.get("should_continue", True):
        return "write_research_brief"
    else:
        return "__end__"

def should_continue_after_execution(state: AgentState) -> str:
    """决定执行后的路径"""
    if state.get("should_end", False):
        return "__end__"
    elif state.get("should_compress", False):
        return "compress_research"
    elif state.get("should_search", False):
        return "perform_searches"
    else:
        return "__end__"

def should_continue_after_analysis(state: AgentState) -> str:
    """决定分析后的路径"""
    if state.get("should_compress", False):
        return "compress_research"
    elif state.get("should_continue_research", False):
        return "plan_research"
    else:
        return "compress_research"  # 默认压缩


# ===== 构建单层图 =====
def build_flat_graph():
    """构建扁平化的单层图"""
    builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
    
    # 添加所有节点到主图级别
    builder.add_node("clarify_with_user", clarify_with_user_node)
    builder.add_node("write_research_brief", write_research_brief_node)
    builder.add_node("generate_queries", generate_queries_node)  # 新增：生成查询节点
    builder.add_node("plan_research", plan_research_node)
    builder.add_node("execute_research_tools", execute_research_tools_node)
    builder.add_node("perform_searches", perform_searches_node)  # 关键节点：执行Tavily搜索
    builder.add_node("analyze_search_results", analyze_search_results_node)
    builder.add_node("compress_research", compress_research_node)
    builder.add_node("generate_final_report", generate_final_report_node)
    
    # 添加边 - 构建完整的工作流
    builder.add_edge(START, "clarify_with_user")
    
    # 使用条件边替代Command控制的路由
    builder.add_conditional_edges(
        "clarify_with_user", 
        should_continue_after_clarify, 
        ["write_research_brief", "__end__"]
    )
    
    builder.add_edge("write_research_brief", "generate_queries")
    builder.add_edge("generate_queries", "plan_research") 
    builder.add_edge("plan_research", "execute_research_tools")
    
    builder.add_conditional_edges(
        "execute_research_tools", 
        should_continue_after_execution, 
        ["perform_searches", "compress_research", "__end__"]
    )
    
    builder.add_edge("perform_searches", "analyze_search_results")
    
    builder.add_conditional_edges(
        "analyze_search_results", 
        should_continue_after_analysis, 
        ["compress_research", "plan_research"]
    )
    
    builder.add_edge("compress_research", "generate_final_report")
    builder.add_edge("generate_final_report", END)
    
    return builder.compile()


# 导出扁平图实例
flat_researcher = build_flat_graph()