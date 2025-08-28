# Flat Graph 前端集成说明

## 概述

`flat_graph.py` 是一个专门为前端集成优化的单层LangGraph架构，解决了原有双层嵌套图架构中事件无法传播到前端的问题。

## 核心改进

### 1. 单层架构
- **问题**：原有的`deep_researcher.py`使用三层嵌套架构（main → supervisor → researcher），导致子图节点事件被封装，前端无法接收
- **解决**：将所有重要节点提升到主图级别，确保事件能直接传播到前端

### 2. 兼容的事件格式
改造后的事件格式完全兼容当前项目前端的`App.tsx`事件处理逻辑：

```javascript
// 前端期望的事件格式
if (event.generate_query) {
  // 查询生成事件
}
if (event.web_research) {
  // 网络研究事件  
}
if (event.reflection) {
  // 反思分析事件
}
if (event.finalize_answer) {
  // 最终报告生成事件
}
```

## 节点映射

| 前端期望事件 | flat_graph节点 | 发送时机 |
|------------|-------------|---------|
| `planner_node` | `plan_research_node` | 开始规划研究任务 |
| `generate_query` | `generate_queries_node` | 生成搜索查询 |
| `web_research` | `perform_searches_node` | 执行Tavily搜索 |
| `reflection` | `analyze_search_results_node` | 分析搜索结果 |
| `record_task_completion` | `execute_research_tools_node` | 记录任务进度 |
| `finalize_answer` | `generate_final_report_node` | 生成最终报告 |

## 使用方法

### 1. 导入改造后的图
```python
from open_deep_research.flat_graph import flat_researcher

# 使用flat_researcher替代原来的deep_researcher
graph = flat_researcher
```

### 2. 配合前端使用
```python
# 在你的API端点中
async def research_stream(query: str):
    config = {
        "configurable": {
            "research_model": "openai:gpt-4o",
            "max_researcher_iterations": 3
        }
    }
    
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=query)]},
        config=config
    ):
        yield chunk  # 这些事件会自动传播到前端
```

### 3. 前端接收事件
当前项目的前端`App.tsx`已经包含了完整的事件处理逻辑，无需修改即可接收flat_graph的事件：

```typescript
onUpdateEvent: (event: any) => {
  if (event.generate_query) {
    // 显示查询生成进度
  } else if (event.web_research) {
    // 显示搜索进度
  } else if (event.reflection) {
    // 显示分析进度  
  } else if (event.finalize_answer) {
    // 显示报告生成进度
  }
}
```

## 事件流程

```
用户输入
↓
clarify_with_user → 澄清需求
↓  
write_research_brief → 编写研究简报
↓
generate_queries → 🚀 发送generate_query事件
↓
plan_research → 🚀 发送planner_node/planner事件
↓
execute_research_tools → 🚀 发送record_task_completion事件
↓
perform_searches → 🚀 发送web_research事件
↓
analyze_search_results → 🚀 发送reflection事件  
↓
compress_research → 压缩研究结果
↓
generate_final_report → 🚀 发送finalize_answer事件
↓
输出最终报告
```

## 关键优势

1. **实时进度显示**：前端可以实时看到每个研究阶段的进展
2. **完整事件覆盖**：包含查询生成、搜索执行、结果分析、报告生成等所有关键步骤
3. **零修改集成**：可以直接替换现有项目的后端，前端无需任何改动
4. **稳定的架构**：单层架构避免了嵌套图的复杂性和事件传播问题

## 测试验证

要验证集成是否成功，可以检查：

1. **前端Activity Timeline组件**应该显示完整的进度步骤
2. **控制台日志**应该显示各个节点的执行信息
3. **最终报告**应该正常生成并显示在前端

## 注意事项

- 确保`.env`文件包含必要的API密钥（TAVILY_API_KEY, AZURE_OPENAI_API_KEY等）
- flat_graph使用与原有图相同的配置格式，无需额外设置
- 如果需要自定义事件格式，可以修改各节点中的`writer()`调用

这个改造让open_deep_research获得了与当前项目相同的优秀前端集成体验！