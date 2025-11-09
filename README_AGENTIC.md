# AI Race Engineer - Production Agentic System

A production-ready **LangGraph-based multi-agent system** for NASCAR racing telemetry analysis. This system uses Claude AI to orchestrate specialized agents that analyze telemetry data and provide setup recommendations.

## 🎯 System Architecture

### Multi-Agent Workflow

```
                    ┌─────────────────────────────┐
                    │    SUPERVISOR AGENT         │
                    │  (Orchestrates workflow)    │
                    └────────────┬────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌──────────────┐         ┌──────────────┐
            │ SPECIALIST   │         │   TOOLS      │
            │   AGENTS     │         │  (12 tools)  │
            └──────────────┘         └──────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌─────────┐ ┌────────┐ ┌─────────────┐
   │  Data   │ │Knowledge│ │   Setup     │
   │ Analyst │ │ Expert  │ │  Engineer   │
   └─────────┘ └────────┘ └─────────────┘
        │           │            │
        └───────────┴────────────┘
                    │
              (Loop back to supervisor)
```

### Agents

1. **Supervisor**: Orchestrates workflow, routes to specialists, synthesizes results
2. **Data Analyst**: Loads telemetry, assesses quality, runs statistical analysis
3. **Knowledge Expert**: Queries setup manuals, searches historical patterns
4. **Setup Engineer**: Generates specific recommendations, validates constraints

### Tools (12 Total)

**Data Operations:**
- `load_telemetry`: Load .ibt, .ldx, or CSV telemetry files
- `inspect_quality`: Assess data quality, detect outliers
- `clean_data`: Remove outliers and prepare data

**Statistical Analysis:**
- `select_features`: Choose relevant parameters based on variance and driver feedback
- `correlation_analysis`: Pearson correlation with lap times
- `regression_analysis`: Multivariate linear regression

**Knowledge & Validation:**
- `query_setup_manual`: Search NASCAR setup knowledge base
- `search_history`: Find similar historical sessions
- `check_constraints`: Validate against driver constraints
- `validate_physics`: Check racing physics principles

**Output:**
- `visualize_impacts`: Generate parameter impact charts
- `save_session`: Persist results to database

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/jacksong17/AI-Race-Engineer.git
cd AI-Race-Engineer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `langgraph`: Multi-agent orchestration
- `langchain-anthropic`: Claude AI integration
- `pandas`, `numpy`, `scipy`: Data analysis
- `scikit-learn`: Machine learning
- `matplotlib`, `seaborn`: Visualization

### 3. Set Up API Key

Create `.env` file with your Anthropic API key:

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

Get your API key from: https://console.anthropic.com/

## 📖 Usage

### Basic Usage

```bash
python main.py --feedback "Car is loose off turn 2, rear end slides on throttle"
```

### With Telemetry Files

```bash
python main.py \
  --feedback "Tight on corner entry, pushes through the middle" \
  --telemetry "data/telemetry/*.csv"
```

### With Constraints

```bash
# Create constraints.json
cat > constraints.json <<EOF
{
  "parameters_at_limit": {
    "tire_psi_rr": "min"
  },
  "cannot_adjust": ["cross_weight"],
  "already_tried": ["track_bar_height_left"]
}
EOF

python main.py \
  --feedback "Still loose on exit" \
  --constraints constraints.json
```

### Verbose Mode

```bash
python main.py \
  --feedback "Oversteer on throttle application" \
  --verbose
```

### Save Session Results

```bash
python main.py \
  --feedback "Understeer in turns 1-2" \
  --save-session \
  --output output/sessions
```

### Display Graph Structure

```bash
python main.py --show-graph
```

## 🔧 How It Works

### Workflow Example

```
1. User provides driver feedback: "Car is loose off turn 2"

2. SUPERVISOR receives input
   → Routes to DATA_ANALYST

3. DATA_ANALYST:
   → Calls load_telemetry()
   → Calls inspect_quality()
   → Calls correlation_analysis()
   → Returns to SUPERVISOR with findings

4. SUPERVISOR synthesizes data
   → Routes to KNOWLEDGE_EXPERT

5. KNOWLEDGE_EXPERT:
   → Calls query_setup_manual("oversteer")
   → Provides setup principles
   → Returns to SUPERVISOR

6. SUPERVISOR decides ready for recommendations
   → Routes to SETUP_ENGINEER

7. SETUP_ENGINEER:
   → Reviews statistical analysis
   → Calls check_constraints()
   → Generates specific recommendation
   → Calls visualize_impacts()
   → Returns to SUPERVISOR

8. SUPERVISOR marks analysis COMPLETE

9. System outputs final recommendation
```

### Iterative Refinement

The supervisor can loop agents multiple times (max 5 iterations) if:
- Initial analysis reveals gaps
- Constraints require alternative recommendations
- Data quality issues need addressing

## 🎓 LangGraph Features Demonstrated

### 1. **State Management**
- Complex `TypedDict` state with 30+ fields
- Message-based agent communication
- State persistence across iterations

### 2. **Conditional Routing**
- Supervisor dynamically routes based on context
- Different paths for different driver complaints
- Iteration limits prevent infinite loops

### 3. **Tool Integration**
- 12 specialized tools bound to agents
- Agents dynamically select which tools to use
- Tool results inform next decisions

### 4. **ReAct Pattern**
- Agents reason about what to do
- Act by calling tools
- Observe results
- Repeat as needed

### 5. **Multi-Agent Coordination**
- Supervisor orchestrates specialists
- Specialists communicate via shared state
- Results synthesized across agents

## 💰 Cost Optimization

**Model Used:** Claude 3 Haiku (`claude-3-haiku-20240307`)

**Why Haiku:**
- Fast response times (< 3 seconds per call)
- Low cost (~$0.25 per 1M input tokens)
- Sufficient quality for routing and tool selection

**Cost Per Analysis:**
- Typical: 3-6 LLM calls
- Total tokens: ~10-15K
- **Estimated cost: $0.03-0.05** ✅

**Compared to alternatives:**
- Sonnet 3.5: ~$0.15-0.25 per analysis (3-5x more expensive)
- GPT-4: ~$0.20-0.30 per analysis (4-6x more expensive)

## 📊 Example Output

```
======================================================================
🎯 SUPERVISOR: Orchestrating analysis workflow
======================================================================

📋 Supervisor Decision:
NEXT_AGENT: data_analyst
REASONING: Need to load and analyze telemetry data first
...

======================================================================
📊 DATA ANALYST: Analyzing telemetry data
======================================================================

🔧 Calling 3 tool(s)...
   → load_telemetry(['data_paths'])
   → inspect_quality(data_dict)
   → correlation_analysis(data_dict, features)

📝 Summary: Analysis complete. Strong negative correlation (-0.42)
            found on tire_psi_rr...

======================================================================
🎯 SUPERVISOR: Orchestrating analysis workflow
======================================================================

📋 Supervisor Decision:
NEXT_AGENT: knowledge_expert
...

======================================================================
📚 KNOWLEDGE_EXPERT: Consulting setup knowledge
======================================================================

🔧 Calling 1 tool(s)...
   → query_setup_manual(issue_type='oversteer')

📝 Summary: Oversteer typically addressed by increasing rear grip...

======================================================================
🎯 SUPERVISOR: Orchestrating analysis workflow
======================================================================

📋 Supervisor Decision:
NEXT_AGENT: setup_engineer
...

======================================================================
🔧 SETUP ENGINEER: Generating recommendations
======================================================================

🔧 Calling 2 tool(s)...
   → check_constraints(parameter='tire_psi_rr', direction='decrease')
   → visualize_impacts(results)

📝 Recommendations: Reduce tire_psi_rr by 1.5 PSI...

======================================================================
📋 FINAL RESULTS
======================================================================

💡 RECOMMENDATION:
  Primary Change:
    • tire_psi_rr
    • Decrease by 1.5 PSI
    • Rationale: Strong negative correlation (-0.42) indicates
      lower pressure will increase mechanical grip at corner exit
    • Confidence: 80%

📊 VISUALIZATIONS:
  • output/visualizations/parameter_impacts_20250109_123456.png

🔄 WORKFLOW STATS:
  • Iterations: 3
  • Agents consulted: data_analyst, knowledge_expert, setup_engineer
  • Tools called: 6
```

## 📁 Project Structure

```
AI-Race-Engineer/
├── race_engineer/          # Core agentic system
│   ├── state.py           # State schema
│   ├── tools.py           # 12 tool implementations
│   ├── agents.py          # Agent node implementations
│   ├── graph.py           # LangGraph workflow
│   └── prompts.py         # Agent system prompts
│
├── data/
│   ├── telemetry/         # Raw telemetry files (.ibt, .ldx, .csv)
│   ├── processed/         # Processed session data
│   └── knowledge/
│       ├── setup_manual.json    # NASCAR setup knowledge base
│       └── checkpoints.db       # LangGraph state persistence
│
├── output/
│   ├── sessions/          # Saved session results
│   └── visualizations/    # Generated charts
│
├── main.py               # CLI interface
├── requirements.txt      # Python dependencies
└── .env                  # API keys (create from .env.example)
```

## 🧪 Development

### Run Tests

```bash
# Test graph structure
python main.py --show-graph

# Test with mock data (no telemetry files needed)
python main.py --feedback "Test feedback"

# Verbose mode for debugging
python main.py --feedback "..." --verbose
```

### Add New Tools

1. Add tool function to `race_engineer/tools.py`
2. Decorate with `@tool`
3. Add to `ALL_TOOLS` list
4. Bind to appropriate agent in `race_engineer/agents.py`

### Add New Agent

1. Create agent prompt in `race_engineer/prompts.py`
2. Implement agent node in `race_engineer/agents.py`
3. Add node to graph in `race_engineer/graph.py`
4. Update supervisor routing logic

## 🎯 Use Cases

### For Racing

- **Pre-race setup**: Analyze test sessions and optimize setup
- **Practice analysis**: Identify issues and recommend changes between sessions
- **Setup tuning**: Systematic parameter optimization
- **Driver feedback integration**: Combine qualitative and quantitative data

### For Interviews/Portfolio

Demonstrates:
- **LangGraph mastery**: StateGraph, routing, tools, checkpointing
- **Agent architecture**: Supervisor pattern, specialized agents
- **Tool integration**: Dynamic tool selection, ReAct pattern
- **Production engineering**: Error handling, cost optimization, logging
- **Domain expertise**: Real motorsports application with physics validation
- **Python best practices**: Type hints, modularity, documentation

## 🔍 Technical Highlights

### State Management
- **Complex state schema** with 30+ fields tracking all analysis aspects
- **Message-based communication** between agents using LangChain messages
- **State persistence** via SQLite checkpointing for long-running workflows

### Routing Logic
- **Dynamic routing** based on supervisor LLM decisions
- **Conditional edges** that adapt to workflow needs
- **Iteration limits** prevent infinite loops
- **Error handling** with graceful degradation

### Tool Calling
- **Intelligent tool selection** - agents choose which tools to use
- **Sequential tool calls** - agents can call multiple tools in sequence
- **Tool result integration** - tool outputs inform next decisions

### Cost Efficiency
- **Haiku model** for 10x cost savings vs Sonnet
- **Selective LLM use** - only supervisor and agents, not tools
- **Parallel tool execution** where possible

## 📚 Resources

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Anthropic Claude**: https://www.anthropic.com/claude
- **NASCAR Setup Guide**: Embedded in `data/knowledge/setup_manual.json`

## 📝 License

MIT License - See LICENSE file

## 👤 Author

Built as a production-grade demonstration of agentic AI systems using LangGraph.

---

**Ready to analyze your racing data!** 🏁
