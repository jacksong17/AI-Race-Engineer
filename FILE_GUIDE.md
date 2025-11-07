# 📁 Bristol AI Race Engineer - File Organization Guide

## Quick Navigation

**Want to run the demo?** → `demo.py`
**Need quick reference?** → `DEMO_CHEATSHEET.txt`
**Preparing for technical audience?** → `TECHNICAL_DEMO.md`
**Testing the system?** → `validate_setup.py`

---

## 📊 File Organization

### 🚀 **Main Application Files** (Run These)

| File | Purpose | When to Use |
|------|---------|-------------|
| **demo.py** ⭐ | Fast, reliable demo (5 seconds) | **Start here** - Recommended for presentations |
| **main.py** | Full demo with visualizations (~30 sec) | When you want charts generated |
| **validate_setup.py** | Test all components | Before presenting to ensure everything works |
| **show_graph.py** | Display workflow graph | To show architecture visually |

**Commands:**
```bash
python demo.py              # Quick demo (recommended)
python main.py              # Full demo with visualizations
python validate_setup.py    # Verify setup
python show_graph.py        # Show graph structure
```

---

### 🧠 **Core Code Files** (The Engine)

| File | Lines | Purpose |
|------|-------|---------|
| **race_engineer.py** | 240 | LangGraph workflow + 3 agents (Telemetry, Analysis, Engineer) |
| **ibt_parser.py** | 350 | Parses native iRacing .ibt binary telemetry files |
| **telemetry_parser.py** | 233 | Parses MoTec .ldx XML setup files |
| **create_visualizations.py** | 266 | Generates dashboard and insight charts |

**Key Lines to Know:**
- `race_engineer.py:17` - TypedDict state definition
- `race_engineer.py:66` - Analysis agent (regression logic)
- `race_engineer.py:163` - Graph construction
- `ibt_parser.py:17` - Graceful degradation pattern

---

### 📚 **Documentation Files** (Read Before Demo)

#### **For Presenting:**

| File | Purpose | Read Time |
|------|---------|-----------|
| **DEMO_CHEATSHEET.txt** ⭐ | Single-page quick reference | 5 min - Keep open during demo |
| **DEMO_WALKTHROUGH.md** | Step-by-step demo script | 15 min - Detailed instructions |
| **TECHNICAL_DEMO.md** | Deep dive for technical audience | 20 min - CS/AI leadership |
| **TECHNICAL_TALKING_POINTS.md** | Key points to memorize | 10 min - "Five technical gems" |

#### **For Understanding:**

| File | Purpose | Read Time |
|------|---------|-----------|
| **README.md** | Project overview, setup, usage | 10 min - Start here if new |
| **PROJECT_STATUS.md** | Current status, what's fixed, ready | 5 min - Quick status check |
| **FRAMEWORK_COMPARISON.md** | Why LangGraph vs alternatives | 10 min - Decision rationale |
| **PRACTICE_NOW.md** | Practice guide with exercises | 15 min - Before presenting |

#### **Historical:**

| File | Purpose |
|------|---------|
| **implementation_plan.md** | Original 5-day plan (reference only) |

---

### 🧪 **Test & Utility Files**

| File | Purpose | When to Use |
|------|---------|-------------|
| **quick_test.py** | Component testing | Quick smoke test |
| **test_parser.py** | Parser testing | When you have .ldx files |
| **requirements.txt** | Python dependencies | `pip install -r requirements.txt` |

---

### 📊 **Data & Visualization Files**

| File | Size | Purpose |
|------|------|---------|
| **trucks silverado2019_bristol...ibt** | 12 MB | Real Bristol telemetry data |
| **bristol_analysis_dashboard.png** | 488 KB | Pre-generated 6-panel dashboard |
| **bristol_key_insights.png** | 165 KB | Pre-generated 3-panel insights |

**Note:** PNG files are pre-generated. Run `create_visualizations.py` to regenerate.

---

### 📁 **Directories**

| Directory | Purpose | Contents |
|-----------|---------|----------|
| **bristol_data/** | Training data storage | Place .ldx files here |
| **data/raw/telemetry/** | Real telemetry files | Contains 1 .ibt file |
| **data/processed/** | Processed datasets | Empty (for future use) |
| **output/** | Results and reports | `demo_results.json` |
| **venv/** | Python virtual environment | Don't touch |
| **.claude/** | Claude Code metadata | Don't touch |

---

## 🎯 File Workflow by Use Case

### **Case 1: Running Demo for First Time**

```
1. README.md           (understand what this is)
2. validate_setup.py   (ensure it works)
3. DEMO_CHEATSHEET.txt (quick reference)
4. demo.py            (run it!)
```

### **Case 2: Preparing for Technical Presentation**

```
1. TECHNICAL_DEMO.md              (read full script)
2. TECHNICAL_TALKING_POINTS.md    (memorize key points)
3. DEMO_CHEATSHEET.txt            (quick reference)
4. PRACTICE_NOW.md                (practice twice)
5. validate_setup.py              (verify before demo)
6. demo.py                        (the actual demo)
```

### **Case 3: Understanding the Code**

```
1. README.md               (overview)
2. race_engineer.py        (agent architecture)
3. ibt_parser.py          (data parsing)
4. FRAMEWORK_COMPARISON.md (design decisions)
```

### **Case 4: Extending the System**

```
1. race_engineer.py:163    (see how to add nodes/edges)
2. race_engineer.py:17     (TypedDict state definition)
3. TECHNICAL_DEMO.md       (extensibility section)
4. Add your agent function
5. Wire it into the graph
```

---

## 📏 File Size Summary

**Total Project Size:** ~13 MB

**Breakdown:**
- Core code: ~50 KB (10 files)
- Documentation: ~110 KB (8 files)
- Telemetry data: 12 MB (1 .ibt file)
- Visualizations: 650 KB (2 PNG files)

---

## 🎯 Essential vs Optional

### **Essential (Can't demo without these):**
- ✅ demo.py
- ✅ race_engineer.py
- ✅ ibt_parser.py
- ✅ telemetry_parser.py
- ✅ requirements.txt
- ✅ DEMO_CHEATSHEET.txt

### **Highly Recommended:**
- ⭐ README.md
- ⭐ TECHNICAL_DEMO.md (if technical audience)
- ⭐ validate_setup.py
- ⭐ Pre-generated PNG charts

### **Optional (Nice to Have):**
- main.py (if you want to generate charts live)
- create_visualizations.py (if you want custom charts)
- show_graph.py (if you want to show graph structure)
- Test files (for development)
- Historical docs (for context)

---

## 🧹 Cleaned Up (Removed)

The following redundant files have been removed:
- ❌ `__pycache__/` - Python cache
- ❌ `DEMO_QUICK_REFERENCE.txt` - Redundant with DEMO_CHEATSHEET.txt
- ❌ `lap_statistics_sample.csv` - Generated sample, not needed
- ❌ `parsed_telemetry_sample.json` - Generated sample, not needed
- ❌ `QUICK_START.md` - Outdated, superseded by README.md

---

## 🗂️ Clean Directory Structure

```
Bristol AI Race Engineer/
├── 🚀 Main Scripts (4 files)
│   ├── demo.py                          ⭐ Start here
│   ├── main.py
│   ├── validate_setup.py
│   └── show_graph.py
│
├── 🧠 Core Code (4 files)
│   ├── race_engineer.py
│   ├── ibt_parser.py
│   ├── telemetry_parser.py
│   └── create_visualizations.py
│
├── 📚 Documentation (8 files)
│   ├── README.md
│   ├── DEMO_CHEATSHEET.txt              ⭐ Quick reference
│   ├── DEMO_WALKTHROUGH.md
│   ├── TECHNICAL_DEMO.md                ⭐ Technical audience
│   ├── TECHNICAL_TALKING_POINTS.md
│   ├── FRAMEWORK_COMPARISON.md
│   ├── PRACTICE_NOW.md
│   ├── PROJECT_STATUS.md
│   └── implementation_plan.md
│
├── 🧪 Tests & Config (3 files)
│   ├── quick_test.py
│   ├── test_parser.py
│   └── requirements.txt
│
├── 📊 Data & Visualizations (3 files)
│   ├── trucks_silverado...ibt           (12 MB - real data)
│   ├── bristol_analysis_dashboard.png
│   └── bristol_key_insights.png
│
└── 📁 Directories (4 folders)
    ├── bristol_data/         (for .ldx training data)
    ├── data/                 (for telemetry storage)
    ├── output/               (for results)
    └── venv/                 (virtual environment)

Total: 22 files + 4 directories
Clean, organized, professional structure
```

---

## 🎯 Quick Commands Reference

```bash
# Navigate to project
cd "C:\Users\jacks\Desktop\AI Race Engineer"

# Run the demo
python demo.py

# Validate everything works
python validate_setup.py

# Show graph structure
python show_graph.py

# Generate visualizations
python create_visualizations.py

# Run full demo with charts
python main.py

# Test components
python quick_test.py
```

---

## ✅ File Checklist for Demo Day

Before your presentation, verify you have:

- [ ] All Python files present (8 total)
- [ ] All documentation files (8 total)
- [ ] Real .ibt telemetry file (12 MB)
- [ ] Pre-generated PNG charts (2 files)
- [ ] `output/` directory exists
- [ ] `validate_setup.py` passes all tests
- [ ] `demo.py` runs successfully
- [ ] `DEMO_CHEATSHEET.txt` printed or open

---

**Your project is now clean, organized, and ready for presentation!** 🏁
