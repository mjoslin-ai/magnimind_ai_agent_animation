import os
import re
import ast
import json
import uuid
import shutil
import logging
import tempfile
import subprocess
from datetime import datetime
from typing import List, Optional, Dict, Any, TypedDict
from pydantic import BaseModel, Field

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Manim AI Agent Generator",
    page_icon="🎬",
    layout="wide"
)

# --- Pydantic Models & State Schema ---
class AnimationPrompt(BaseModel):
    description: str = Field(..., description="Topic or concept to animate")
    complexity: str = Field("medium", description="Complexity level: simple, medium, or complex")
    quality: str = Field("medium_quality", description="Render quality")
    target_duration: float = Field(90.0, description="Desired total video length in seconds")

class AnimationScenario(BaseModel):
    title: str = Field(..., description="Title of the animation")
    objects: List[str] = Field(..., description="Visual elements/objects to include")
    transformations: List[str] = Field(..., description="Animations and transitions to apply")
    equations_or_text: Optional[List[str]] = Field(None, description="Key text or equations to feature")
    storyboard: Optional[List[Dict[str, Any]]] = Field(None, description="Step-by-step storyboard sections")

class AnimationState(TypedDict):
    prompt: str
    complexity: str
    quality: str
    scenario: Optional[Dict[str, Any]]
    raw_code: str
    optimized_code: str
    final_code: str
    video_path: Optional[str]
    is_valid: bool
    error_message: Optional[str]
    retry_count: int
    target_duration: float

# --- Helper Function ---
def _extract_text(content) -> str:
    """Helper to safely convert LLM response content (string or list) into a string."""
    if isinstance(content, list):
        return "".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content])
    return str(content)

# --- LangGraph Nodes ---
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)

def extract_scenario_node(state: AnimationState) -> Dict[str, Any]:
    logger.info(f"--- NODE: Extract Scenario for topic: {state['prompt']} ---")
    llm = get_llm()
    prompt = state["prompt"]
    complexity = state["complexity"]
    
    system_prompt = """
    Create an educational animation storyboard for the given topic (can be mathematics, science, computer science, data structures, physics, or general concepts).
    Focus on clear visual storytelling.
    Respond with a JSON object containing:
    - title: A clear, engaging title
    - objects: Visual elements/objects to include (e.g., shapes, text labels, nodes, graphs, icons)
    - transformations: Animation types to use (e.g., "fade_in", "transform", "shift", "grow_arrow")
    - equations_or_text: Key text elements or formulas to feature (can be null)
    - storyboard: 4-6 chronological sections, each with:
      * section_name: Name of section
      * time_range: Estimated timing
      * narration: Explanation text
      * visuals: What appears on screen
      * animations: Specific transitions
    The ENTIRE animation must total approximately {target_duration} seconds. Distribute the time evenly (or with natural pacing) across the sections.
    Only respond with the valid JSON object, nothing else.
    """
    
    messages = [
        ("system", system_prompt.format(target_duration=state["target_duration"])),
        ("user", f"Create an animation storyboard for the topic: '{prompt}'. "
                f"Complexity: {complexity}. Target total duration: {state['target_duration']} seconds.")
    ]
    
    response = llm.invoke(messages)
    content_str = _extract_text(response.content)
    
    try:
        json_match = re.search(r'\{.*\}', content_str, re.DOTALL)
        if json_match:
            return {"scenario": json.loads(json_match.group(0))}
    except Exception as e:
        logger.error(f"Error parsing scenario JSON: {e}")
        
    return {
        "scenario": {
            "title": f"{prompt.capitalize()} Overview",
            "objects": ["text_labels", "shapes", "containers"],
            "transformations": ["fade_in", "move_to"],
            "equations_or_text": None
        }
    }

def generate_code_node(state: AnimationState) -> Dict[str, Any]:
    logger.info("--- NODE: Generate Manim Code ---")
    llm = get_llm()
    scenario = state["scenario"]
    prompt = state["prompt"]
    complexity = state["complexity"]
    
    system_prompt = """
    Create professional Manim animation code in Python that visualizes the requested topic clearly and elegantly. Your code MUST:
    1. Use 'from manim import *' at the top.
    2. Create a Scene class named 'ManimScene' extending Scene.
    3. Implement the construct method correctly.
    4. CRITICAL CONSTANTS & API RULES:
       - DO NOT use 'CENTER'. Use 'ORIGIN' for centering objects.
       - Text objects do not take an 'alignment' argument. For center-aligned paragraphs, use Paragraph(..., alignment="center").
    5. Explicitly position elements using move_to(), shift(), or to_edge() to avoid overlaps.
    6. Clean up old elements between sections using self.play(FadeOut(...)) (DO NOT use self.clear()).
    7. Return ONLY valid Python code inside ```python markdown blocks. No extra conversational filler.
    8. Use ONLY objects, methods and constants that exist in the official Manim Community API.
    9. CRITICAL TIMING RULE:
        - The total length of the animation (sum of all run_time arguments + self.wait calls)
            MUST be approximately {target_duration} seconds.
        - Prefer explicit run_time=... on every self.play() and self.wait().
        - Typical pacing: 8-15 s per major section for a {target_duration}s video.
        - Do NOT leave large silent gaps; keep the viewer engaged for the full duration.
    """
    
    messages = [
        ("system", system_prompt.format(target_duration=state["target_duration"])),
        ("user", f"Topic: {prompt}\nTitle: {scenario.get('title')}\n"
                 f"Objects: {scenario.get('objects')}\n"
                 f"Transformations: {scenario.get('transformations')}\n"
                 f"Complexity: {complexity}\n"
                 f"Target duration: {state['target_duration']} seconds\n"
                 f"Storyboard: {scenario.get('storyboard')}")
    ]
    
    response = llm.invoke(messages)
    content = _extract_text(response.content)
    
    if "```python" in content:
        content = content.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]
        
    return {"raw_code": content.strip()}

def optimize_layout_node(state: AnimationState) -> Dict[str, Any]:
    logger.info("--- NODE: Optimize Layout ---")
    llm = get_llm()
    code = state["raw_code"]
    
    system_prompt = """
    Review the provided Manim code and ensure optimal layout and flow:
    1. Check for proper spacing (at least 2.0 units between active elements).
    2. Ensure proper cleanup of past elements using FadeOut before introducing new concepts.
    3. Return ONLY the polished Python code inside ```python markdown blocks.
    """
    
    messages = [
        ("system", system_prompt),
        ("user", f"Code to optimize:\n```python\n{code}\n```")
    ]
    
    response = llm.invoke(messages)
    content = _extract_text(response.content)
    
    if "```python" in content:
        content = content.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]
        
    return {"optimized_code": content.strip(), "final_code": content.strip()}

def validate_code_node(state: AnimationState) -> Dict[str, Any]:
    logger.info("--- NODE: Validate Manim Code (AST & Rules Check) ---")
    code = state["final_code"]
    errors = []
    
    try:
        ast.parse(code)
    except SyntaxError as e:
        errors.append(f"SyntaxError: {e.msg} at line {e.lineno}")
        
    if "from manim import *" not in code:
        errors.append("Missing required import: 'from manim import *'")
        
    if "class " not in code or "Scene" not in code:
        errors.append("Missing Manim Scene class definition (e.g., class MyScene(Scene):)")
        
    if "self.clear()" in code:
        errors.append("Forbidden method call detected: 'self.clear()' is not supported; use explicit FadeOut animations.")
        
    if "CENTER" in code:
        errors.append("Invalid constant 'CENTER': Manim uses 'ORIGIN' for the center point.")
        
    if errors:
        error_msg = "; ".join(errors)
        logger.warning(f"Validation failed: {error_msg}")
        return {"is_valid": False, "error_message": error_msg}
    else:
        logger.info("Code validation passed successfully.")
        return {"is_valid": True, "error_message": None}

def fix_code_node(state: AnimationState) -> Dict[str, Any]:
    logger.info("--- NODE: Fix Manim Code ---")
    llm = get_llm()
    code = state["final_code"]
    error_msg = state.get("error_message") or "Unknown validation or runtime error."
    retry_count = state.get("retry_count", 0) + 1
    
    system_prompt = """
    You are an expert Manim and Python debugging assistant.
    Fix the provided Manim code by addressing all identified issues. Follow these guidelines:
    1. Fix all syntax errors and logical mistakes first
    2. Fix positioning issues by adding explicit positioning commands
    3. Resolve element overlaps by repositioning elements with adequate spacing
    4. Maintain the original educational intent and mathematical content
    5. Ensure all animations follow a logical step-by-step flow
    Return ONLY the corrected, valid Python code inside ```python markdown blocks.
    """
    
    messages = [
        ("system", system_prompt),
        ("user", f"Code with error:\n```python\n{code}\n```\n\nValidation Errors / Feedback:\n{error_msg}")
    ]
    
    response = llm.invoke(messages)
    content = _extract_text(response.content)
    
    if "```python" in content:
        content = content.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]
        
    return {
        "final_code": content.strip(),
        "optimized_code": content.strip(),
        "retry_count": retry_count
    }

def should_fix_code(state: AnimationState) -> str:
    if not state.get("is_valid", True) and state.get("retry_count", 0) < 3:
        return "fix"
    return "proceed"

def render_manim_video(code: str, quality: str = "medium_quality") -> str:
    base_temp = os.path.abspath("./tmp_manim")
    output_dir = os.path.abspath("./videos")
    os.makedirs(base_temp, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    short_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join(base_temp, short_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    script_path = os.path.join(temp_dir, "script.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
        
    class_name = "ManimScene"
    for line in code.split("\n"):
        if line.strip().startswith("class ") and "Scene" in line:
            class_name = line.strip().split("class ")[1].split("(")[0].strip()
            break
            
    quality_flags = {
        "low_quality": ("-ql", "480p15"),
        "medium_quality": ("-qm", "720p30"),
        "high_quality": ("-qh", "1080p60")
    }
    flag, quality_dir = quality_flags.get(quality, ("-qm", "720p30"))
    
    command = ["manim", flag, "script.py", class_name]
    logger.info(f"Executing command: {' '.join(command)} in {temp_dir}")
    
    result = subprocess.run(
        command, 
        cwd=temp_dir, 
        capture_output=True, 
        text=True, 
        encoding="utf-8", 
        errors="replace"
    )
    
    if result.returncode != 0:
        logger.error(f"Manim render error: {result.stderr}")
        return f"Error: {result.stderr}"
        
    videos_dir = os.path.join(temp_dir, "media", "videos")
    if not os.path.exists(videos_dir):
        return "Error: Video output directory not found."
        
    scene_dirs = [os.path.join(videos_dir, d) for d in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, d))]
    if not scene_dirs:
        return "Error: Scene directory not found."
        
    scene_dir = max(scene_dirs, key=os.path.getctime)
    target_dir = os.path.join(scene_dir, quality_dir)
    if not os.path.exists(target_dir):
        target_dir = scene_dir
        
    mp4_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".mp4")]
    if not mp4_files:
        return "Error: No MP4 file generated."
        
    video_file = max(mp4_files, key=os.path.getctime)
    timestamp = datetime.now().strftime("%m%d%H%M")
    output_file = os.path.join(output_dir, f"vid_{timestamp}_{short_id}.mp4")
    
    shutil.copy2(video_file, output_file)
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return output_file

def render_video_node(state: AnimationState) -> Dict[str, Any]:
    logger.info("--- NODE: Render Video & Check Runtime ---")
    code = state["final_code"]
    quality = state.get("quality", "medium_quality")
    
    video_path = render_manim_video(code, quality=quality)
    
    if video_path.startswith("Error:"):
        error_msg = video_path.replace("Error:", "").strip()
        retry_count = state.get("retry_count", 0) + 1
        logger.warning(f"Runtime render failed: {error_msg}")
        return {
            "is_valid": False,
            "error_message": f"RuntimeError/NameError during Manim execution: {error_msg}",
            "retry_count": retry_count,
            "video_path": None
        }
    else:
        logger.info("Video rendered successfully!")
        return {
            "is_valid": True,
            "error_message": None,
            "video_path": video_path
        }

@st.cache_resource
def get_compiled_workflow():
    workflow = StateGraph(AnimationState)
    workflow.add_node("extract_scenario", extract_scenario_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("optimize_layout", optimize_layout_node)
    workflow.add_node("validate_code", validate_code_node)
    workflow.add_node("fix_code", fix_code_node)
    workflow.add_node("render_video", render_video_node)
    
    workflow.add_edge(START, "extract_scenario")
    workflow.add_edge("extract_scenario", "generate_code")
    workflow.add_edge("generate_code", "optimize_layout")
    workflow.add_edge("optimize_layout", "validate_code")
    
    workflow.add_conditional_edges(
        "validate_code",
        should_fix_code,
        {"fix": "fix_code", "proceed": "render_video"}
    )
    workflow.add_conditional_edges(
        "render_video",
        should_fix_code,
        {"fix": "fix_code", "proceed": END}
    )
    workflow.add_edge("fix_code", "validate_code")
    return workflow.compile()

# --- Streamlit UI Layout ---
st.title("🎬 Manim AI Animation Agent")
st.markdown("Transform any topic into a professional math or computer science animation using **LangGraph**, **Gemini**, and **Manim**.")

with st.sidebar:
    st.header("🔑 Configuration")
    api_key_input = st.text_input("Google Gemini API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input
        
    st.divider()
    st.header("⚙️ Animation Settings")
    complexity_choice = st.selectbox("Complexity", ["simple", "medium", "complex"], index=1)
    quality_choice = st.selectbox("Render Quality", ["low_quality", "medium_quality", "high_quality"], index=1)
    target_dur = st.slider("Target Duration (seconds)", min_value=30.0, max_value=120.0, value=90.0, step=10.0)

# Main Form Input
topic_input = st.text_input("Enter Topic or Concept to Animate:", placeholder="e.g., How Dijkstra's Algorithm Works")

if st.button("🚀 Generate Animation", type="primary"):
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Please enter a valid Google Gemini API Key in the sidebar or environment variables.")
    elif not topic_input.strip():
        st.warning("Please enter a valid animation topic.")
    else:
        app_graph = get_compiled_workflow()
        
        initial_state = {
            "prompt": topic_input,
            "complexity": complexity_choice,
            "target_duration": target_dur,
            "quality": quality_choice,
            "scenario": None,
            "raw_code": "",
            "optimized_code": "",
            "final_code": "",
            "video_path": "",
            "is_valid": True,
            "error_message": None,
            "retry_count": 0
        }
        
        with st.status("🤖 Running Multi-Agent Workflow...", expanded=True) as status:
            st.write("📋 Extracting storyboard and script plan...")
            # Invoke app graph
            try:
                result = app_graph.invoke(initial_state)
                status.update(label="✅ Animation Generation Complete!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Workflow execution failed: {e}")
                result = None
                
        if result:
            scenario = result.get("scenario", {})
            st.subheader(f"📌 {scenario.get('title', topic_input)}")
            
            # Display storyboard breakdown if available
            if scenario.get("storyboard"):
                with st.expander("📖 View Storyboard Plan"):
                    for idx, sec in enumerate(scenario.get("storyboard", [])):
                        st.markdown(f"**Section {idx+1}: {sec.get('section_name')}** ({sec.get('time_range')})")
                        st.write(f"*Narration:* {sec.get('narration')}")
                        st.write(f"*Visuals:* {sec.get('visuals')}")
                        st.divider()

            video_path = result.get("video_path")
            if video_path and os.path.exists(video_path):
                st.success("Video successfully rendered!")
                st.video(video_path)
                
                with open(video_path, "rb") as file:
                    st.download_button(
                        label="📥 Download MP4 Animation",
                        data=file,
                        file_name=os.path.basename(video_path),
                        mime="video/mp4"
                    )
            else:
                st.error("Failed to render video. Check logs for details.")
                
            with st.expander("💻 View Generated Manim Code"):
                st.code(result.get("final_code", "# No code generated"), language="python")