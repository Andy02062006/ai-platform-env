import gradio as gr
from env import AIPlatformEnv
from models import Action
import json

# Initialize environment
env = AIPlatformEnv(seed=42)

# Custom CSS for "Classy" look
custom_css = """
footer {visibility: hidden}
.gradio-container {background-color: #0f172a !important; color: #f8fafc !important;}
.main-row {background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 20px; border: 1px solid #334155;}
.label-text {color: #94a3b8 !important; font-weight: 600;}
.response-card {
    border: 1px solid #334155; 
    padding: 15px; 
    border-radius: 8px; 
    background: #1e293b; 
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.response-card:hover {border-color: #38bdf8; transform: translateY(-2px);}
.badge {
    padding: 2px 8px; 
    border-radius: 12px; 
    font-size: 0.75rem; 
    font-weight: bold;
}
.badge-rel {background: #0369a1; color: #7dd3fc;}
.badge-conf {background: #1e3a8a; color: #93c5fd;}
"""

def format_responses(responses):
    if not responses:
        return "<p style='color: #64748b; text-align: center; padding: 20px;'>Wait for observations...</p>"
    
    html = "<div style='display: flex; flex-direction: column; gap: 10px;'>"
    for i, r in enumerate(responses):
        html += f"""
        <div class='response-card'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <span style='font-weight: bold; color: #f1f5f9;'>Candidate {i}</span>
                <span style='display: flex; gap: 8px;'>
                    <span class='badge badge-rel'>Relevance: {r.relevance:.2f}</span>
                    <span class='badge badge-conf'>Confidence: {r.confidence:.2f}</span>
                </span>
            </div>
            <p style='margin: 0; color: #cbd5e1; line-height: 1.5;'>{r.text}</p>
        </div>
        """
    html += "</div>"
    return html

def step_env(action_type, query, selected_index, score, plan):
    try:
        if action_type == "submit_query":
            action = Action(type="submit_query", query=query)
        elif action_type == "select_response":
            action = Action(type="select_response", selected_index=int(selected_index))
        elif action_type == "rate_response":
            action = Action(type="rate_response", score=float(score))
        elif action_type == "refine_query":
            action = Action(type="refine_query", query=query)
        elif action_type == "plan_task":
            action = Action(type="plan_task", plan=plan)
        elif action_type == "compare_responses":
            action = Action(type="compare_responses")
        elif action_type == "summarize":
            action = Action(type="summarize")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        history_text = "\n".join(obs.history)
        resp_html = format_responses(obs.responses)
        
        status = "Active"
        status_color = "#38bdf8"
        if terminated: 
            status = "Completed"
            status_color = "#4ade80"
        if truncated: 
            status = "Max Turns Reached"
            status_color = "#f87171"
        
        total_score_html = f"<div style='font-size: 2.5rem; font-weight: 800; color: #38bdf8;'>{info['total_reward']:.2f}</div>"
        last_reward_html = f"<div style='font-size: 1.2rem; font-weight: 600; color: #94a3b8;'>+{reward.value:.2f}</div>"
        
        return (
            resp_html, 
            history_text, 
            total_score_html,
            last_reward_html,
            f"Turn {info['turns_used']} / {info['max_turns']}",
            f"<span style='color: {status_color}; font-weight: bold;'>{status}</span>"
        )
    except Exception as e:
        return f"Error: {str(e)}", "Error", "N/A", "N/A", "N/A", "Error"

def reset_env(task_key):
    obs, info = env.reset(task_key)
    return (
        format_responses(obs.responses),
        "",
        "<div style='font-size: 2.5rem; font-weight: 800; color: #38bdf8;'>0.00</div>",
        "<div style='font-size: 1.2rem; font-weight: 600; color: #94a3b8;'>0.00</div>",
        f"Turn 0 / {info['max_turns']}",
        "<span style='color: #38bdf8; font-weight: bold;'>Initialized</span>"
    )

# Build UI
# Reverting theme and css to Blocks for wider compatibility
with gr.Blocks(title="AIPlatformEnv | Premium Lab", theme=gr.themes.Default(), css=custom_css) as demo:
    gr.Markdown("""
    # AIPlatformEnv Pro
    **Environment Lab** | Strategic Interaction & Quality Benchmarking
    """)
    
    with gr.Row(elem_classes=["main-row"]):
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### Session Config")
                task_select = gr.Dropdown(
                    choices=["easy", "medium", "hard"], 
                    value="easy", 
                    label="Complexity Level"
                )
                reset_btn = gr.Button("Initialize New Session", variant="secondary")
            
            with gr.Group():
                gr.Markdown("### Strategic Action")
                action_type = gr.Radio(
                    choices=[
                        "submit_query", "refine_query", "plan_task", 
                        "compare_responses", "rate_response", "select_response"
                    ],
                    value="submit_query",
                    label="Action Protocol"
                )
                
                with gr.Column(visible=True) as query_col:
                    query_input = gr.Textbox(label="Query / Refinement", placeholder="Enter strategic input...")
                    plan_input = gr.Textbox(label="Conceptual Plan", placeholder="Describe architecture...")
                
                with gr.Row():
                    idx_input = gr.Number(label="Target Candidate", value=0, precision=0)
                    score_input = gr.Slider(label="Calibrated Score", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                
                step_btn = gr.Button("Execute Action", variant="primary")

        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Cumulative Reward")
                    total_display = gr.HTML("<div style='font-size: 2.5rem; font-weight: 800; color: #38bdf8;'>0.00</div>")
                with gr.Column():
                    gr.Markdown("#### Last Signal")
                    reward_display = gr.HTML("<div style='font-size: 1.2rem; font-weight: 600; color: #94a3b8;'>0.00</div>")
                with gr.Column():
                    gr.Markdown("#### Progress")
                    turn_display = gr.Markdown("Turn 0 / -")
                    status_display = gr.HTML("<span style='color: #64748b;'>Idle</span>")
            
            with gr.Tabs():
                with gr.TabItem("Observations"):
                    responses_display = gr.HTML(value=format_responses([]))
                with gr.TabItem("Event Log"):
                    history_display = gr.Textbox(label=None, lines=15, interactive=False, container=False)

    # Event handlers
    reset_btn.click(
        reset_env, 
        inputs=[task_select], 
        outputs=[responses_display, history_display, total_display, reward_display, turn_display, status_display]
    )
    
    step_btn.click(
        step_env,
        inputs=[action_type, query_input, idx_input, score_input, plan_input],
        outputs=[responses_display, history_display, total_display, reward_display, turn_display, status_display]
    )

if __name__ == "__main__":
    # Simplified launch() to avoid argument errors
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860
    )
