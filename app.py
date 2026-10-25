"""
Resume Roast AI
A modern, AI-powered resume critique web application powered by Google Gemma 4 (gemma-4-26b-a4b-it).
Built for MLH Open-Source AI Hackathon.
"""

import os
import json
import re
from typing import Tuple, Optional, Dict, Any
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Constants
MODEL_ID = "gemma-4-26b-a4b-it"
DEFAULT_TARGET_ROLE = "Entry-Level Software Engineer"

SAMPLE_RESUMES = {
    "Weak Junior Dev": (
        "Alex Johnson\n"
        "alex.j@email.com | 555-0199 | github.com/alexj\n\n"
        "OBJECTIVE:\n"
        "Looking for a software developer job at a good tech company where I can use my skills and learn new things.\n\n"
        "EDUCATION:\n"
        "State University - B.S. in Computer Science (2020 - 2024)\n"
        "GPA: 2.9/4.0\n\n"
        "SKILLS:\n"
        "Python, Java, C++, HTML, CSS, JavaScript, React, Node.js, SQL, MongoDB, Git, Docker, Kubernetes, AWS, Machine Learning, AI\n\n"
        "PROJECTS:\n"
        "E-Commerce Website\n"
        "- Made an e-commerce website using MERN stack.\n"
        "- Created login and registration pages.\n"
        "- Connected MongoDB database to store user information and products.\n"
        "- Used CSS for styling.\n\n"
        "Weather App\n"
        "- Built a weather application in React.\n"
        "- Called an external weather API to fetch temperature and forecasts.\n"
        "- Hosted the project on Vercel.\n\n"
        "WORK EXPERIENCE:\n"
        "Campus IT Support Desk Assistant (2022 - 2024)\n"
        "- Helped students reset their campus passwords.\n"
        "- Fixed computer hardware and software issues on campus lab PCs.\n"
        "- Answered support tickets and emails.\n"
    ),
    "College Intern / CS Student": (
        "Morgan Lee\n"
        "morgan.lee@cs.univ.edu | linkedin.com/in/morganlee-dev\n\n"
        "EDUCATION:\n"
        "Tech Institute of Technology - B.S. Computer Science (Expected May 2025)\n"
        "Relevant Coursework: Data Structures, Algorithms, Operating Systems, Database Systems\n\n"
        "TECHNICAL SKILLS:\n"
        "Languages: Python, Go, TypeScript, SQL\n"
        "Tools: Docker, PostgreSQL, Redis, Git, Linux, GitHub Actions\n\n"
        "EXPERIENCE:\n"
        "Software Engineering Intern - CloudFlow Inc. (June 2024 - August 2024)\n"
        "- Worked with the backend team on microservices in Go.\n"
        "- Wrote unit tests for API endpoints.\n"
        "- Participated in daily standups and sprint planning.\n"
        "- Helped migrate old database queries to PostgreSQL.\n\n"
        "PROJECTS:\n"
        "Distributed Task Queue (Go, Redis)\n"
        "- Created a lightweight background job processing queue in Go.\n"
        "- Implemented worker concurrency with goroutines and Redis pub/sub.\n"
        "- Added exponential backoff retry mechanism for failed jobs.\n\n"
        "StudyGroup Matcher (Python, FastAPI, SQLite)\n"
        "- Built a REST API backend allowing students to form collaborative study sessions.\n"
        "- Implemented JWT authentication and automated swagger documentation.\n"
    ),
}


# ==========================================
# Core Helper Functions (Pure Logic & Testing)
# ==========================================

def get_api_key() -> Optional[str]:
    """
    Retrieve Gemini API key from environment, Streamlit secrets, or session state.
    Never exposes or hardcodes the key.
    """
    # 1. Check environment variable (from .env or OS environment)
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 2. Check Streamlit secrets (for cloud deployments)
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"].strip():
            return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        pass

    # 3. Check Streamlit session state (user manual input in sidebar)
    if "manual_api_key" in st.session_state and st.session_state["manual_api_key"]:
        return st.session_state["manual_api_key"].strip()

    return None


def validate_inputs(resume_text: str, target_role: str, api_key: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate input parameters before sending to Gemma inference.
    Returns (is_valid, error_message).
    """
    if not api_key:
        return False, "Gemini API key is missing. Please set `GEMINI_API_KEY` in your `.env` file or enter it in the sidebar."

    if not target_role or not target_role.strip():
        return False, "Target role cannot be empty. Please specify the role you are targeting."

    clean_resume = resume_text.strip() if resume_text else ""
    if not clean_resume:
        return False, "Please paste your resume text before roasting."

    # Minimum length checks to prevent empty/insufficient roasts
    if len(clean_resume) < 80:
        return False, f"Resume text is too brief ({len(clean_resume)} characters). Please paste at least 80 characters containing your experience, skills, or projects."

    word_count = len(clean_resume.split())
    if word_count < 15:
        return False, f"Resume contains only {word_count} words. Please provide a more complete resume snippet for an accurate roast."

    return True, None


def get_score_badge(score: int) -> Dict[str, str]:
    """
    Return appropriate grading label, fire level, and color for a score out of 100.
    """
    if score >= 90:
        return {"grade": "A+", "status": "Interview Magnet", "color": "#10b981", "flame": "🔥 Ready to Hire"}
    elif score >= 80:
        return {"grade": "A-", "status": "Solid Contender", "color": "#34d399", "flame": "⚡ Light Sear"}
    elif score >= 70:
        return {"grade": "B", "status": "Decent Foundation", "color": "#f59e0b", "flame": "♨️ Medium Rare"}
    elif score >= 55:
        return {"grade": "C", "status": "Needs Serious Work", "color": "#ff6b35", "flame": "🌶️ Well Done"}
    elif score >= 40:
        return {"grade": "D", "status": "Resume in Distress", "color": "#ef4444", "flame": "💥 Extra Crispy"}
    else:
        return {"grade": "F", "status": "Total Meltdown", "color": "#dc2626", "flame": "🌋 Incinerated"}


def build_roast_prompt(resume_text: str, target_role: str, roast_intensity: str = "Extra Crispy") -> str:
    """
    Construct the precise structured prompt for Gemma (gemma-4-26b-a4b-it).
    Enforces strict integrity guidelines (no hallucinated metrics) and strict JSON output.
    """
    tone_instructions = {
        "Gentle Simmer": "Tone: Warm, encouraging, gently playful, but honest about weaknesses.",
        "Medium Roast": "Tone: Sharp, humorous, tech-savvy, witty, and realistically critical.",
        "Extra Crispy": "Tone: Brutally witty, hilarious, savage tech roast humor, but fundamentally constructive and high-value.",
    }.get(roast_intensity, "Tone: Brutally witty, tech roast humor, yet genuinely constructive.")

    prompt = f"""You are 'Resume Roast AI', a notoriously witty, sharp-tongued Principal Software Engineering Recruiter and Tech Lead.
Your mission is to perform a brutally honest, constructive roast of the candidate's resume for their target role: '{target_role}'.

{tone_instructions}

============================================================
CRITICAL INTEGRITY INSTRUCTIONS (MANDATORY):
1. NEVER fabricate, invent, or hallucinate achievements, company names, certifications, credentials, or metrics that are not in the resume.
2. If metrics or quantifiable outcomes are missing from the candidate's original resume, you MUST use bracketed placeholders like `[X%]`, `[number]`, `[latency decrease]`, or `[tool name]` so the candidate knows what authentic data they must calculate and fill in.
3. Be specific to the candidate's actual text — cite exact weaknesses and vague phrasing.
============================================================

TARGET ROLE: {target_role}

CANDIDATE'S RESUME:
\"\"\"
{resume_text}
\"\"\"

You MUST respond ONLY with a single valid JSON object. Do NOT enclose your output in markdown codeblocks (no ```json or ```). Do NOT include any conversational preamble or outro text.
The JSON must adhere strictly to the following schema:

{{
  "overall_score": <integer from 0 to 100 representing overall quality for the target role>,
  "score_breakdown": {{
    "clarity": <integer 0-100 evaluating formatting, conciseness, readability, and structure>,
    "skills": <integer 0-100 evaluating relevance, currency, and depth of technical skills for {target_role}>,
    "project_descriptions": <integer 0-100 evaluating depth, problem-solving narrative, and technical architecture of projects>,
    "measurable_impact": <integer 0-100 evaluating use of metrics, quantifiable results, business value, and scale>
  }},
  "roast_punchline": "<A one-sentence witty, punchy, hilarious roast summary that delivers the burn of the resume>",
  "verdict_summary": "<A 3-4 sentence constructive, witty roast-style assessment detailing where the candidate stands, what is holding them back from landing {target_role}, and why fixing it is urgent>",
  "weaknesses": [
    {{
      "title": "<Concise name of Major Weakness 1>",
      "explanation": "<Detailed explanation of why this weakens the resume for {target_role}>",
      "impact": "<How recruiters or ATS filters react to this specific flaw>"
    }},
    {{
      "title": "<Concise name of Major Weakness 2>",
      "explanation": "<Detailed explanation of why this weakens the resume for {target_role}>",
      "impact": "<How recruiters or ATS filters react to this specific flaw>"
    }},
    {{
      "title": "<Concise name of Major Weakness 3>",
      "explanation": "<Detailed explanation of why this weakens the resume for {target_role}>",
      "impact": "<How recruiters or ATS filters react to this specific flaw>"
    }}
  ],
  "rewritten_bullets": [
    {{
      "original": "<Exact or representative weak bullet point from the candidate's resume>",
      "rewritten": "<Powerful, action-verb driven, STAR-method rewrite with [X%]/[metric] placeholders for missing numbers>",
      "why": "<Brief explanation of why this rewrite works better>"
    }},
    {{
      "original": "<Exact or representative weak bullet point from the candidate's resume>",
      "rewritten": "<Powerful, action-verb driven, STAR-method rewrite with [X%]/[metric] placeholders for missing numbers>",
      "why": "<Brief explanation of why this rewrite works better>"
    }},
    {{
      "original": "<Exact or representative weak bullet point from the candidate's resume>",
      "rewritten": "<Powerful, action-verb driven, STAR-method rewrite with [X%]/[metric] placeholders for missing numbers>",
      "why": "<Brief explanation of why this rewrite works better>"
    }}
  ],
  "prioritized_improvements": [
    {{
      "priority": 1,
      "area": "<Key Area 1>",
      "action": "<Concrete, immediate actionable step the candidate must take>"
    }},
    {{
      "priority": 2,
      "area": "<Key Area 2>",
      "action": "<Concrete, immediate actionable step the candidate must take>"
    }},
    {{
      "priority": 3,
      "area": "<Key Area 3>",
      "action": "<Concrete, immediate actionable step the candidate must take>"
    }},
    {{
      "priority": 4,
      "area": "<Key Area 4>",
      "action": "<Concrete, immediate actionable step the candidate must take>"
    }},
    {{
      "priority": 5,
      "area": "<Key Area 5>",
      "action": "<Concrete, immediate actionable step the candidate must take>"
    }}
  ]
}}"""
    return prompt


def parse_roast_response(response_text: str) -> Dict[str, Any]:
    """
    Parse and sanitize the Gemma model's response into a robust dictionary.
    Handles potential markdown fences (```json ... ```) and applies safe fallbacks.
    """
    cleaned = response_text.strip()

    # Strip markdown code blocks if the model wrapped them
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Find first '{' and last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = cleaned[first_brace : last_brace + 1]
    else:
        json_candidate = cleaned

    try:
        data = json.loads(json_candidate)
    except Exception as e:
        # Fallback dictionary if JSON parsing fails
        return {
            "overall_score": 50,
            "score_breakdown": {
                "clarity": 50,
                "skills": 50,
                "project_descriptions": 50,
                "measurable_impact": 50,
            },
            "roast_punchline": "Your resume threw our JSON parser off the rails — just like it might confuse an ATS scanner!",
            "verdict_summary": f"The AI analysis completed, but the formatting required a safety recovery. Raw response preview: {cleaned[:300]}...",
            "weaknesses": [
                {
                    "title": "Formatting & Parseability",
                    "explanation": "The resume content or structure produced an unconventional output schema.",
                    "impact": "ATS scanners may struggle to index key sections.",
                },
                {
                    "title": "Impact & Metric Vagueness",
                    "explanation": "Bullet points lack distinct, measurable metrics and outcomes.",
                    "impact": "Leaves recruiters guessing about your direct contributions.",
                },
                {
                    "title": "Project Depth",
                    "explanation": "Projects need clearer architectural depth and technology context.",
                    "impact": "Doesn't fully showcase your readiness for the target role.",
                },
            ],
            "rewritten_bullets": [
                {
                    "original": "Worked on backend features and fixed bugs.",
                    "rewritten": "Architected [number] backend endpoints using [framework], reducing latency by [X%] across [number] users.",
                    "why": "Introduces concrete action verbs and placeholders for real metrics.",
                },
                {
                    "original": "Built frontend using React and CSS.",
                    "rewritten": "Developed responsive, accessible web UI components with React, improving Lighthouse performance score to [X/100].",
                    "why": "Highlights specific engineering outcomes and performance benchmarks.",
                },
                {
                    "original": "Helped manage database and team tasks.",
                    "rewritten": "Optimized database queries and schemas, decreasing average response time by [X%] across [number] tables.",
                    "why": "Eliminates passive phrasing ('helped') in favor of ownership verbs.",
                },
            ],
            "prioritized_improvements": [
                {"priority": 1, "area": "Quantifiable Metrics", "action": "Add specific numbers, percentage increases, and scale to every bullet point."},
                {"priority": 2, "area": "Action Verbs", "action": "Replace passive verbs ('helped', 'worked on') with authoritative verbs ('engineered', 'deployed')."},
                {"priority": 3, "area": "Target Role Alignment", "action": "Audit your listed skills to directly reflect requirements for your target role."},
                {"priority": 4, "area": "Project Architecture", "action": "Explain the 'Why' and technical trade-offs behind your key project designs."},
                {"priority": 5, "area": "Formatting Simplicity", "action": "Maintain clean, single-column formatting for seamless ATS parsing."},
            ],
            "_raw_error": str(e),
        }

    # Ensure required top-level fields are present and sanitized
    def clamp_score(val: Any, default: int = 50) -> int:
        try:
            return max(0, min(100, int(val)))
        except (ValueError, TypeError):
            return default

    overall_score = clamp_score(data.get("overall_score", 50))
    breakdown = data.get("score_breakdown", {})
    if not isinstance(breakdown, dict):
        breakdown = {}

    sanitized_breakdown = {
        "clarity": clamp_score(breakdown.get("clarity", overall_score)),
        "skills": clamp_score(breakdown.get("skills", overall_score)),
        "project_descriptions": clamp_score(breakdown.get("project_descriptions", overall_score)),
        "measurable_impact": clamp_score(breakdown.get("measurable_impact", overall_score)),
    }

    weaknesses = data.get("weaknesses", [])
    if not isinstance(weaknesses, list) or len(weaknesses) == 0:
        weaknesses = [
            {"title": "Missing Measurable Outcomes", "explanation": "Achievements lack numbers and tangible evidence.", "impact": "High rejection risk."},
            {"title": "Generic Technical Scope", "explanation": "Technologies listed without showing how they solved problems.", "impact": "Overlooked by recruiters."},
            {"title": "Passive Language", "explanation": "Phrasing lacks leadership and direct technical ownership.", "impact": "Perceived as junior."},
        ]

    rewritten_bullets = data.get("rewritten_bullets", [])
    if not isinstance(rewritten_bullets, list) or len(rewritten_bullets) == 0:
        rewritten_bullets = [
            {
                "original": "Worked on features and bug fixes.",
                "rewritten": "Designed and shipped [number] core features using [framework], reducing customer-reported bugs by [X%].",
                "why": "Focuses on quantifiable outcome and technology ownership.",
            }
        ]

    improvements = data.get("prioritized_improvements", [])
    if not isinstance(improvements, list) or len(improvements) == 0:
        improvements = [
            {"priority": 1, "area": "Impact", "action": "Add metrics to all experience bullets."},
            {"priority": 2, "area": "Skills", "action": "Prune outdated skills and emphasize primary stack."},
            {"priority": 3, "area": "Projects", "action": "Highlight architectural challenges overcome."},
            {"priority": 4, "area": "Formatting", "action": "Streamline sections for clear ATS parsing."},
            {"priority": 5, "area": "Headline", "action": "Craft a sharp, relevant summary header."},
        ]

    return {
        "overall_score": overall_score,
        "score_breakdown": sanitized_breakdown,
        "roast_punchline": str(data.get("roast_punchline", "A resume with ambition, but crying out for metrics and focus!")),
        "verdict_summary": str(data.get("verdict_summary", "Review completed. Focus on quantifying your technical impact.")),
        "weaknesses": weaknesses[:3],  # Ensure top 3
        "rewritten_bullets": rewritten_bullets[:3],  # Ensure top 3
        "prioritized_improvements": improvements[:5],  # Ensure top 5
    }


def call_gemma_api(prompt: str, api_key: str) -> str:
    """
    Execute real Gemma inference using the Google Gen AI Python SDK.
    Uses model ID 'gemma-4-26b-a4b-it' accessed via the Gemini API client.
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )

    if not response or not hasattr(response, "text") or not response.text:
        raise ValueError("Received empty or invalid response from Gemma model.")

    return response.text


def format_markdown_report(data: Dict[str, Any], target_role: str) -> str:
    """
    Format the complete roast report as a clean, downloadable Markdown document.
    """
    score = data["overall_score"]
    badge = get_score_badge(score)
    bd = data["score_breakdown"]

    lines = [
        f"# 🔥 Resume Roast AI Report",
        f"**Target Role:** {target_role}  ",
        f"**Model:** Google Gemma 4 (`{MODEL_ID}`)  ",
        f"**Overall Score:** {score}/100 — *{badge['status']}* ({badge['flame']})\n",
        f"---",
        f"## 🎙️ The Roast Punchline",
        f"> *\"{data['roast_punchline']}\"*\n",
        f"### ⚖️ The Verdict",
        f"{data['verdict_summary']}\n",
        f"---",
        f"## 📊 Score Breakdown",
        f"- **Clarity & Structure:** {bd['clarity']}/100",
        f"- **Skills Relevance:** {bd['skills']}/100",
        f"- **Project Descriptions:** {bd['project_descriptions']}/100",
        f"- **Measurable Impact:** {bd['measurable_impact']}/100\n",
        f"---",
        f"## ⚠️ 3 Major Weaknesses",
    ]

    for idx, w in enumerate(data["weaknesses"], 1):
        lines.append(f"### {idx}. {w.get('title', 'Weakness')}")
        lines.append(f"**Explanation:** {w.get('explanation', '')}  ")
        lines.append(f"**ATS/Recruiter Impact:** {w.get('impact', '')}\n")

    lines.append("---\n## ✏️ 3 Bullet Point Makeovers")
    for idx, b in enumerate(data["rewritten_bullets"], 1):
        lines.append(f"### Makeover #{idx}")
        lines.append(f"- **Before (Weak):** `{b.get('original', '')}`")
        lines.append(f"- **After (Gemma Rewrite):** **{b.get('rewritten', '')}**")
        lines.append(f"- *Why it works:* {b.get('why', '')}\n")

    lines.append("---\n## 🎯 5 Prioritized Improvements")
    for item in data["prioritized_improvements"]:
        lines.append(f"**Priority {item.get('priority', '')} [{item.get('area', '')}]:** {item.get('action', '')}")

    lines.append("\n---\n*Notice: This score and roast is an AI-generated estimate generated by Google Gemma (`gemma-4-26b-a4b-it`) for educational and self-improvement purposes, not an official hiring guarantee.*")

    return "\n".join(lines)


# ==========================================
# Streamlit Custom Styling & UI Injector
# ==========================================

CUSTOM_CSS = """
<style>
/* Main Dark Background & Typography */
.stApp {
    background-color: #0c0b14;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Header Gradient */
.roast-header {
    background: linear-gradient(135deg, #ff6b35 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}

.roast-subheader {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 1.8rem;
}

/* Badge pill */
.badge-pill {
    display: inline-block;
    background: rgba(168, 85, 247, 0.15);
    border: 1px solid rgba(168, 85, 247, 0.4);
    color: #c084fc;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.35rem 0.8rem;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
}

/* Glassmorphism Cards */
.roast-card {
    background: rgba(22, 20, 37, 0.7);
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
}

.roast-card-accent {
    border-left: 4px solid #ff6b35;
}

.roast-card-purple {
    border-left: 4px solid #a855f7;
}

/* Hero Score Box */
.score-hero-box {
    text-align: center;
    background: radial-gradient(circle at center, rgba(168, 85, 247, 0.15) 0%, rgba(255, 107, 53, 0.08) 70%, rgba(12, 11, 20, 0.8) 100%);
    border: 1px solid rgba(255, 107, 53, 0.4);
    border-radius: 18px;
    padding: 2rem 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 35px rgba(255, 107, 53, 0.15);
}

.score-number {
    font-size: 4.8rem;
    font-weight: 900;
    line-height: 1;
    background: linear-gradient(135deg, #ff6b35 0%, #f59e0b 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.5rem 0;
}

.score-grade {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.4rem 1.2rem;
    border-radius: 8px;
    display: inline-block;
    margin-top: 0.4rem;
}

/* Sub-score grid */
.metric-pill {
    background: rgba(30, 27, 48, 0.8);
    border: 1px solid rgba(168, 85, 247, 0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}

.metric-pill-title {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}

.metric-pill-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f8fafc;
}

/* Bullet makeover Before/After blocks */
.before-block {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #fca5a5;
    font-family: monospace;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}

.after-block {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #a7f3d0;
    font-family: inherit;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* Action button glow */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #ff6b35 0%, #e11d48 50%, #7928ca 100%);
    color: #ffffff;
    font-weight: 700;
    font-size: 1.15rem;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    width: 100%;
    box-shadow: 0 4px 20px rgba(255, 107, 53, 0.35);
    transition: all 0.3s ease;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(255, 107, 53, 0.55);
    background: linear-gradient(135deg, #ff7b00 0%, #f43f5e 50%, #8a2be2 100%);
}

/* Text area styling */
.stTextArea textarea {
    background-color: #13121f !important;
    border: 1px solid rgba(168, 85, 247, 0.3) !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
}

.stTextArea textarea:focus {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 10px rgba(255, 107, 53, 0.3) !important;
}

/* Input fields */
.stTextInput input {
    background-color: #13121f !important;
    border: 1px solid rgba(168, 85, 247, 0.3) !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}

/* Notice box */
.disclaimer-box {
    background: rgba(30, 27, 48, 0.6);
    border: 1px solid rgba(255, 107, 53, 0.3);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #cbd5e1;
    margin-top: 2rem;
    text-align: center;
}
</style>
"""


# ==========================================
# Main Streamlit Application UI
# ==========================================

def main():
    st.set_page_config(
        page_title="Resume Roast AI | Google Gemma 4",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject dark neon theme
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.markdown("<div class='badge-pill'>MLH Hackathon • Open Source</div>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: #ff6b35; margin-top: 0;'>⚙️ Settings & Info</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            **Model:** `{MODEL_ID}`  
            **Architecture:** Google Gemma 4 MoE  
            **Provider:** Google Gen AI SDK (`google-genai`)
            """
        )

        st.divider()

        # API Key Configuration
        st.markdown("### 🔑 API Authentication")
        detected_key = get_api_key()

        if detected_key:
            st.success("API Key detected & active! ✅")
        else:
            st.warning("No API Key found in `.env` or secrets.")

        manual_key_input = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("manual_api_key", ""),
            type="password",
            help="Free key available from Google AI Studio. Stored in memory only.",
            placeholder="AIzaSy...",
        )
        if manual_key_input:
            st.session_state["manual_api_key"] = manual_key_input

        st.markdown(
            "[👉 Get a free Gemini API Key](https://aistudio.google.com/app/apikey)",
            unsafe_allow_html=True,
        )

        st.divider()

        # Roast Intensity
        st.markdown("### 🌶️ Roasting Intensity")
        roast_intensity = st.radio(
            "Select Heat Level:",
            options=["Medium Roast", "Extra Crispy", "Gentle Simmer"],
            index=1,
            help="Controls the wit and spice level of the critique while keeping advice constructive.",
        )

        st.divider()

        # Quick Sample Resumes
        st.markdown("### 📋 Sample Resumes")
        st.caption("Click to instantly load a sample resume into the text area:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("Weak Junior", use_container_width=True):
                st.session_state["resume_input_text"] = SAMPLE_RESUMES["Weak Junior Dev"]
                st.session_state["target_role_input"] = "Entry-Level Software Engineer"
                st.rerun()
        with col_s2:
            if st.button("CS Intern", use_container_width=True):
                st.session_state["resume_input_text"] = SAMPLE_RESUMES["College Intern / CS Student"]
                st.session_state["target_role_input"] = "Backend Engineer Intern"
                st.rerun()

        if st.button("🧹 Clear All Inputs", use_container_width=True):
            st.session_state["resume_input_text"] = ""
            st.session_state["target_role_input"] = DEFAULT_TARGET_ROLE
            st.session_state.pop("roast_result", None)
            st.rerun()

        st.divider()
        st.caption("Powered by Google Gemma 4 (`gemma-4-26b-a4b-it`) • Built for MLH Hackathon.")

    # Main Page Header
    st.markdown("<div style='text-align: center;'><span class='badge-pill'>Gemma 4 MoE Powered • MLH Hackathon</span></div>", unsafe_allow_html=True)
    st.markdown("<h1 class='roast-header'>🔥 Resume Roast AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='roast-subheader'>Turn your resume from a bland list of tasks into an ATS-crushing, interview-landing powerhouse.</p>", unsafe_allow_html=True)

    # Input Section
    col_input1, col_input2 = st.columns([2, 1])

    with col_input1:
        target_role = st.text_input(
            "🎯 Target Job Role",
            value=st.session_state.get("target_role_input", DEFAULT_TARGET_ROLE),
            placeholder="e.g. Entry-Level Software Engineer, Frontend Developer, Data Scientist",
            help="The job title you want the AI to benchmark your resume against.",
            key="target_role_input",
        )

    with col_input2:
        st.markdown("<label style='font-size: 0.85rem; color: #94a3b8; font-weight: 500;'>Quick Role Presets</label>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("Frontend Dev", use_container_width=True):
                st.session_state["target_role_input"] = "Junior Frontend Developer"
                st.rerun()
        with col_r2:
            if st.button("Data / ML", use_container_width=True):
                st.session_state["target_role_input"] = "Entry-Level Machine Learning Engineer"
                st.rerun()

    # Large Resume Text Area
    resume_text = st.text_area(
        "📄 Paste Your Resume Text",
        value=st.session_state.get("resume_input_text", ""),
        height=260,
        placeholder="Paste your plain-text resume here (experience, skills, projects, education)...\n\nTip: You can use the 'Sample Resumes' buttons in the sidebar for an instant demo!",
        key="resume_input_text",
    )

    # Word & Char Count Display
    clean_text = resume_text.strip() if resume_text else ""
    c_count = len(clean_text)
    w_count = len(clean_text.split()) if clean_text else 0

    col_meta1, col_meta2 = st.columns([1, 1])
    with col_meta1:
        if c_count > 0:
            st.caption(f"📊 Stats: **{w_count} words** • **{c_count} characters**")
        else:
            st.caption("Waiting for resume input...")
    with col_meta2:
        st.caption("🔒 Privacy: Your resume is processed in real time and never stored on any server.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Prominent Roast Button
    roast_clicked = st.button("🔥 Roast My Resume", use_container_width=True)

    active_api_key = get_api_key()

    if roast_clicked:
        # Validate inputs
        is_valid, error_msg = validate_inputs(resume_text, target_role, active_api_key)

        if not is_valid:
            st.error(f"⚠️ {error_msg}")
        else:
            prompt = build_roast_prompt(resume_text, target_role, roast_intensity)

            with st.spinner("🔥 Preheating the grill and consulting Google Gemma 4 (`gemma-4-26b-a4b-it`)..."):
                try:
                    raw_response = call_gemma_api(prompt, active_api_key)
                    parsed_result = parse_roast_response(raw_response)
                    st.session_state["roast_result"] = parsed_result
                    st.session_state["last_role"] = target_role
                    st.toast("🔥 Your resume has been roasted! Scroll down to see results.", icon="🌶️")
                except Exception as ex:
                    st.error(f"❌ An error occurred during inference: {str(ex)}")
                    st.info(
                        "💡 **Troubleshooting Tips:**\n"
                        "1. Verify that your `GEMINI_API_KEY` is valid in `.env` or the sidebar.\n"
                        "2. Ensure you have network connectivity to Google's API.\n"
                        "3. Confirm that the model `gemma-4-26b-a4b-it` is accessible from your API key's region."
                    )

    # Display Roast Results
    if "roast_result" in st.session_state:
        result = st.session_state["roast_result"]
        current_target = st.session_state.get("last_role", target_role)

        st.markdown("<br><hr style='border-color: rgba(168, 85, 247, 0.3);'><br>", unsafe_allow_html=True)

        # Hero Score Box
        overall_score = result["overall_score"]
        badge = get_score_badge(overall_score)
        breakdown = result["score_breakdown"]

        st.markdown(
            f"""
            <div class='score-hero-box'>
                <div class='badge-pill'>AI Resume Evaluation • {current_target}</div>
                <div style='color: #94a3b8; font-size: 1.1rem; text-transform: uppercase; font-weight: 600;'>Overall Resume Score</div>
                <div class='score-number'>{overall_score}<span style='font-size: 2rem; color: #94a3b8;'>/100</span></div>
                <div class='score-grade' style='background: {badge["color"]}22; color: {badge["color"]}; border: 1px solid {badge["color"]}66;'>
                    {badge["grade"]} • {badge["status"]} ({badge["flame"]})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sub-Scores Row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f"""
                <div class='metric-pill'>
                    <div class='metric-pill-title'>Clarity & Structure</div>
                    <div class='metric-pill-value' style='color: #38bdf8;'>{breakdown.get("clarity", 0)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(breakdown.get("clarity", 0) / 100)

        with col_m2:
            st.markdown(
                f"""
                <div class='metric-pill'>
                    <div class='metric-pill-title'>Skills Match</div>
                    <div class='metric-pill-value' style='color: #a855f7;'>{breakdown.get("skills", 0)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(breakdown.get("skills", 0) / 100)

        with col_m3:
            st.markdown(
                f"""
                <div class='metric-pill'>
                    <div class='metric-pill-title'>Project Descriptions</div>
                    <div class='metric-pill-value' style='color: #ff6b35;'>{breakdown.get("project_descriptions", 0)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(breakdown.get("project_descriptions", 0) / 100)

        with col_m4:
            st.markdown(
                f"""
                <div class='metric-pill'>
                    <div class='metric-pill-title'>Measurable Impact</div>
                    <div class='metric-pill-value' style='color: #f43f5e;'>{breakdown.get("measurable_impact", 0)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(breakdown.get("measurable_impact", 0) / 100)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎙️ The Roast & Verdict",
            "⚠️ 3 Critical Weaknesses",
            "✏️ Bullet Makeovers",
            "🎯 5 Prioritized Actions",
            "📥 Export Full Report",
        ])

        # TAB 1: Roast & Verdict
        with tab1:
            st.markdown("<div class='roast-card roast-card-accent'>", unsafe_allow_html=True)
            st.markdown("### 🔥 The Roast Punchline")
            st.markdown(f"> *\"{result.get('roast_punchline', '')}\"*")
            st.markdown("### ⚖️ Recruiter's Verdict")
            st.write(result.get("verdict_summary", ""))
            st.markdown("</div>", unsafe_allow_html=True)

        # TAB 2: Major Weaknesses
        with tab2:
            st.markdown("### ⚠️ Top 3 Red Flags on Your Resume")
            st.caption("Here is what causes recruiters to hit 'Reject' before getting to your portfolio:")

            for i, w in enumerate(result.get("weaknesses", []), 1):
                st.markdown(
                    f"""
                    <div class='roast-card roast-card-accent'>
                        <h4 style='color: #ff6b35; margin-top: 0;'>#{i}: {w.get('title', 'Weakness')}</h4>
                        <p><strong>Why It Hurts You:</strong> {w.get('explanation', '')}</p>
                        <p style='color: #f87171;'><strong>🚨 ATS / Recruiter Consequence:</strong> {w.get('impact', '')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # TAB 3: Bullet Point Makeovers
        with tab3:
            st.markdown("### ✏️ 3 Bullet Point Makeovers")
            st.caption(
                "Notice: To adhere strictly to integrity rules, missing metrics are marked with placeholders like "
                "<code>[X%]</code> or <code>[metric]</code> so you can fill in your real numbers without fabricating facts."
            )

            for i, bullet in enumerate(result.get("rewritten_bullets", []), 1):
                st.markdown(f"#### Makeover #{i}")
                st.markdown(
                    f"""
                    <div class='before-block'>
                        <span style='color: #ef4444; font-weight: bold;'>❌ BEFORE:</span> {bullet.get('original', '')}
                    </div>
                    <div class='after-block'>
                        <span style='color: #10b981; font-weight: bold;'>✨ GEMMA REWRITE:</span> {bullet.get('rewritten', '')}
                    </div>
                    <p style='font-size: 0.9rem; color: #cbd5e1; margin-top: 0.3rem;'>
                        <strong>Why this lands interviews:</strong> {bullet.get('why', '')}
                    </p>
                    <hr style='border-color: rgba(168, 85, 247, 0.15);'>
                    """,
                    unsafe_allow_html=True,
                )

        # TAB 4: Prioritized Improvements
        with tab4:
            st.markdown("### 🎯 5-Step Action Roadmap")
            st.caption("Follow these prioritized steps in order to maximize your callback rate:")

            for item in result.get("prioritized_improvements", []):
                p_num = item.get("priority", 1)
                st.markdown(
                    f"""
                    <div class='roast-card roast-card-purple'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <h4 style='color: #c084fc; margin: 0;'>Priority {p_num} • {item.get('area', 'Improvement')}</h4>
                        </div>
                        <p style='margin-top: 0.6rem; font-size: 1rem;'>{item.get('action', '')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # TAB 5: Export Full Report
        with tab5:
            st.markdown("### 📥 Download & Save Your Roast Report")
            st.caption("Export the complete Gemma-generated analysis as a Markdown file for offline review or sharing.")

            markdown_report = format_markdown_report(result, current_target)

            st.download_button(
                label="📄 Download Full Roast Report (.md)",
                data=markdown_report,
                file_name=f"Resume_Roast_Report_{current_target.replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

            with st.expander("👁️ Preview Raw Markdown Report"):
                st.code(markdown_report, language="markdown")

    # Mandatory Legal / Educational Disclaimer
    st.markdown(
        """
        <div class='disclaimer-box'>
            ⚠️ <strong>Notice:</strong> This score and roast is an AI-generated estimate powered by Google Gemma 4
            (<code>gemma-4-26b-a4b-it</code>). It is intended solely for educational, resume-refinement, and entertainment purposes.
            It does not constitute an official hiring prediction, ATS certification, or employment guarantee.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
