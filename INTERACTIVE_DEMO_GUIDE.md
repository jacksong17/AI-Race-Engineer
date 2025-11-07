# Interactive Demo Guide: AI Race Engineer

**NEW: LLM-Powered Driver Feedback Interpretation**

Your AI Race Engineer now accepts natural language driver feedback and uses AI to interpret it!

---

## 🎯 Why This Matters for Your Demo

**Before (Hardcoded):**
- Driver feedback was pre-programmed
- Same analysis every time
- Looked like a static pipeline

**After (Interactive + LLM):**
- ✅ **Type ANY driver complaint** - system adapts in real-time
- ✅ **LLM interprets** natural language → structured diagnosis
- ✅ **Different inputs** → different agent decisions → different results
- ✅ **Proves agentic behavior** - not just sequential processing!

---

## 🚀 Three Ways to Run the Demo

### Option 1: Interactive Mode (Best for Live Demo)

```bash
python demo.py
```

The system will prompt you:

```
[1.5/5] Driver Feedback Session...

🏁 DRIVER DEBRIEF:
   (The crew chief asks the driver about the car's handling...)

   Enter driver feedback (or press Enter for default):
   Examples:
     - 'Car feels loose coming off the corners'
     - 'Front end pushes in turn 1 and 2'
     - 'Bottoming out in the center of corners'

   Driver: █
```

**Type your feedback and press Enter!**

### Option 2: Command-Line Argument

```bash
# Example 1: Loose car
python demo.py "Car feels loose coming off turn 2, rear end wants to come around"

# Example 2: Tight car
python demo.py "Front end pushes in turns 1 and 2, won't turn in"

# Example 3: Bottoming out
python demo.py "Suspension is bottoming out in the center of corners"

# Example 4: General optimization
python demo.py "Car feels pretty good, just looking for more speed"
```

### Option 3: Default Example (Press Enter)

Just press Enter when prompted, and it uses the default loose car scenario.

---

## 🤖 LLM-Powered vs Rule-Based Interpretation

### With LLM API (Anthropic Claude or OpenAI GPT)

**Setup:**
```bash
# Option A: Anthropic Claude (recommended)
export ANTHROPIC_API_KEY="your-api-key-here"
pip install anthropic

# Option B: OpenAI GPT
export OPENAI_API_KEY="your-api-key-here"
pip install openai
```

**What happens:**
- LLM reads driver's natural language feedback
- Understands nuanced descriptions
- Extracts technical diagnosis
- Identifies priority parameters

**Example:**
```
Driver: "Car is super loose on throttle in 1 and 2, but tight mid-corner in 3 and 4"

LLM Interpretation:
✓ Complaint: mixed_handling
✓ Severity: moderate
✓ Diagnosis: "Complex handling balance issue - loose on power application but tight mid-corner"
✓ Priority: ["tire_psi_rr", "cross_weight", "tire_psi_lf"]
```

### Without LLM API (Rule-Based Fallback)

**What happens:**
- System uses keyword matching
- Still functional and deterministic
- Simpler interpretation

**Example:**
```
Driver: "Car feels loose off corners"

Rule-Based Interpretation:
✓ Complaint: loose_exit (detected "loose" + "corner")
✓ Severity: moderate
✓ Diagnosis: "Insufficient rear grip causing oversteer on throttle application"
✓ Priority: ["tire_psi_rr", "tire_psi_lr", "track_bar_height_left"]
```

**Note:** Rule-based mode still works great for the demo! LLM just adds extra sophistication.

---

## 🎬 Demo Script: Showing Agentic Flexibility

**Key Talking Points:**

### 1. Setup (30 seconds)
"I'm going to show you how the system adapts to ANY driver feedback in real-time. Watch how different inputs lead to different agent decisions."

### 2. First Run - Loose Car (2 minutes)
```bash
python demo.py "Car feels loose off corners, rear end wants to come around"
```

**Point out:**
- Agent 1 diagnoses: "Oversteer (loose rear end)"
- Agent 1 prioritizes: tire_psi_rr, tire_psi_lr
- Agent 2 marks these as [PRIORITY] in analysis
- Agent 3 validates: "Data confirms driver complaint"
- **Result:** Reduce tire_psi_rr

### 3. Second Run - Tight Car (2 minutes)
```bash
python demo.py "Front end pushes in turn 1 and 2"
```

**Point out:**
- Agent 1 diagnoses: "Understeer (tight front end)"
- Agent 1 prioritizes: tire_psi_lf, tire_psi_rf, cross_weight
- Agent 2 focuses on DIFFERENT parameters
- Agent 3 recommends DIFFERENT changes
- **Result:** Adjust front tire pressure / cross weight

### 4. Key Insight (30 seconds)
"Notice how the SAME code produced DIFFERENT analysis based on driver input? That's true agentic behavior:
- **Perception:** LLM understood natural language
- **Reasoning:** Agent 1 connected symptoms to causes
- **Planning:** Agent 2 adapted its analysis
- **Action:** Agent 3 generated context-specific recommendations
- **Not a pipeline** - agents made different decisions based on context!"

---

## 🧪 Test Cases for Your Demo

### Easy Cases (Clear Complaints)
```bash
python demo.py "Loose off corners"
python demo.py "Tight on entry"
python demo.py "Bottoming out"
```

### Complex Cases (Show LLM Reasoning)
```bash
python demo.py "Car is perfect on fresh tires but gets loose after 10 laps"
python demo.py "Turn 1 feels great but turn 3 is super tight"
python demo.py "I'm fighting the car the entire lap, feels unbalanced"
```

### Edge Cases (Show Robustness)
```bash
python demo.py "Just looking to optimize lap time"
python demo.py "Everything feels good"
python demo.py "Not sure what's wrong, just feels off"
```

---

## 📊 What the Audience Sees

### With Loose Feedback:
```
[AGENT 1] Telemetry Chief: Interpreting driver feedback...
   🎧 Driver complaint: 'loose_exit' during corner_exit
   💡 DIAGNOSIS: Oversteer (loose rear end)
   ✓ DECISION: Prioritize REAR GRIP parameters
      Priority features: tire_psi_rr, tire_psi_lr, track_bar_height_left

[AGENT 2] Data Scientist: Selecting analysis strategy...
   🎯 Agent 1 identified priority areas: Oversteer (loose rear end)
   ✓ tire_psi_rr               (varied: σ=0.51) [PRIORITY]
   ✓ tire_psi_lr               (varied: σ=0.71) [PRIORITY]

   Results:
      • tire_psi_rr              : +0.551 [PRIORITY - matches driver feedback]

[AGENT 3] Crew Chief: Synthesizing recommendations...
   ✅ VALIDATION: Top parameter matches driver feedback!
   STRONG RECOMMENDATION: Reduce tire_psi_rr
   🎧 Addresses driver complaint: Oversteer (loose rear end)
```

### With Tight Feedback:
```
[AGENT 1] Telemetry Chief: Interpreting driver feedback...
   🎧 Driver complaint: 'tight_entry' during corner_entry
   💡 DIAGNOSIS: Understeer (tight front end)
   ✓ DECISION: Prioritize FRONT GRIP parameters
      Priority features: tire_psi_lf, tire_psi_rf, cross_weight

[AGENT 2] Data Scientist: Selecting analysis strategy...
   🎯 Agent 1 identified priority areas: Understeer (tight front end)
   ✓ tire_psi_lf               (varied: σ=0.97) [PRIORITY]
   ✓ cross_weight              (varied: σ=1.31) [PRIORITY]

   Results:
      • cross_weight             : -0.289 [PRIORITY - matches driver feedback]

[AGENT 3] Crew Chief: Synthesizing recommendations...
   ✅ VALIDATION: Top parameter matches driver feedback!
   RECOMMENDATION: Increase cross_weight
   🎧 Addresses driver complaint: Understeer (tight front end)
```

**Same Bristol data, different recommendations based on driver input!**

---

## 💡 Presentation Tips

### Opening
"Let me show you something cool - I can type ANY driver complaint, and the agents will adapt their analysis in real-time."

### During Demo
"Watch Agent 1 - it's using an LLM to understand natural language, then reasoning about which technical parameters relate to the driver's complaint."

### The "Wow" Moment
"Now watch what happens when I give it the OPPOSITE complaint..."
*(Run second demo with tight feedback)*
"See how Agent 2 focused on completely different parameters? Same code, same data, different reasoning path. That's what makes this agentic."

### Closing
"This same pattern - natural language input → reasoning → adaptive analysis → validated recommendations - applies to supply chain, healthcare, finance, anywhere you need AI to combine human expertise with data analysis."

---

## 🔧 Troubleshooting

**"No LLM API key"**
- That's fine! System falls back to rule-based interpretation
- Still demonstrates adaptive behavior
- Just slightly less sophisticated

**"Want to use OpenAI instead of Anthropic"**
- Edit `demo.py` line 135: change `llm_provider="openai"`
- Set `OPENAI_API_KEY` environment variable

**"Want to force rule-based (no API calls)"**
- Edit `demo.py` line 135: change `llm_provider="mock"`
- Useful for offline demos

---

## 🎯 Key Messages for Your Presentation

1. **Flexibility**: "Not hardcoded - accepts ANY driver input"
2. **Adaptability**: "Different inputs → different agent decisions"
3. **LLM + Determinism**: "LLM for understanding, math for recommendations"
4. **True Agents**: "Agents reason and plan, not just execute steps"
5. **Production-Ready**: "Graceful fallback when LLM unavailable"

---

## 📝 Quick Reference

```bash
# Interactive (best for live demo)
python demo.py

# Command-line (best for scripted demo)
python demo.py "your driver feedback here"

# Test the interpreter alone
python driver_feedback_interpreter.py
```

**API Keys (optional):**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

Good luck with your demo! 🏁
