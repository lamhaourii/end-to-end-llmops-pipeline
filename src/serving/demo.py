import os
import json
import httpx
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8080")

def summarize_stream(prompt: str, temperature: float, max_tokens: int):
    if not prompt.strip():
        yield "Please enter some text"
        return
    payload = {
        "prompt":      prompt.strip(),
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    summary = ""
    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream(
                "POST",
                f"{API_URL}/generate/stream",
                json=payload
            ) as response:
                if response.status_code != 200:
                    yield f"API Error: {response.status_code}"
                    return
                for line in response.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        raw = line[6:]
                        try:
                            data = json.loads(raw)
                            if "error" in data:
                                yield f"{data['error']}"
                                return
                            if data.get("done"):
                                total_ms     = data.get("total_ms", 0)
                                total_tokens = data.get("total_tokens", 0)
                                summary += f"\n\n---\n⏱️ {total_ms:.0f}ms | 🔤 {total_tokens} tokens"
                                yield summary
                                return
                            if "token" in data:
                                summary += data["token"]
                                yield summary  
                        except json.JSONDecodeError:
                            continue
    except httpx.ConnectError:
        yield f"Cannot connect to API at {API_URL}\nMake sure the API server is running."
    except httpx.TimeoutException:
        yield "Request timed out. Try a shorter prompt or fewer tokens."
    except Exception as e:
        yield f"Unexpected error: {str(e)}"

def check_api_health() -> str:
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        data = response.json()
        if data.get("vllm_healthy"):
            return f"API is online | Model: {data['model']} | Version: {data['version']}"
        else:
            return f"API is up but vLLM is not ready yet"
    except Exception:
        return f"API is offline at {API_URL}"

EXAMPLES = [
    [
        "أعلنت وزارة التربية الوطنية عن إطلاق برنامج جديد لتحسين جودة التعليم في المدارس العمومية المغربية، يشمل توفير أجهزة لوحية لجميع تلاميذ السنة الأولى إعدادي.",
        0.5, 80
    ],
    [
        "سجل الاقتصاد المغربي نمواً بنسبة 3.2 بالمئة خلال الربع الثالث من السنة الجارية، مدفوعاً بالأداء الجيد لقطاعي الفلاحة والسياحة وارتفاع تحويلات المغاربة المقيمين بالخارج.",
        0.5, 80
    ],
    [
        "حقق المنتخب المغربي لكرة القدم فوزاً مهماً على نظيره الغاني بهدفين مقابل هدف ضمن مباريات التصفيات الإفريقية المؤهلة لكأس العالم في ظل أداء متميز من جميع اللاعبين.",
        0.5, 100
    ],
]

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald",
    ),
    title="DarijaLLM — Moroccan Arabic Summarizer",
    css="""
        .rtl-text textarea { direction: rtl; font-family: 'Arial', sans-serif; font-size: 16px; }
        .rtl-output textarea { direction: rtl; font-family: 'Arial', sans-serif; font-size: 16px; }
        .title-block { text-align: center; padding: 20px; }
        footer { display: none !important; }
    """
) as demo:

    with gr.Row(elem_classes="title-block"):
        gr.Markdown("""
        #DarijaLLM — Moroccan Arabic Summarizer
        ### نموذج ذكاء اصطناعي لتلخيص النصوص بالدارجة المغربية

        **English:** A finetuned LLaMA 1B model that summarizes Arabic news articles into Moroccan Darija.  
        **بالدارجة:** نموذج مدرب على الدارجة المغربية باش يلخص لك الأخبار بطريقة مفهومة.

        > Model: `LLaMA-3.2-1B` finetuned with QLoRA on Moroccan Darija news data
        """)
    with gr.Row():
        api_status = gr.Textbox(
            value=check_api_health,
            label="API Status",
            interactive=False,
            every=30,   
        )
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Arabic Text",
                placeholder="Enter Arabic text to summarize",
                lines=8,
                max_lines=15,
                elem_classes="rtl-text",
            )

            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    label="Temperature",
                    info="Low = more focused | High = more creative"
                )
                max_tokens_slider = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=80,
                    step=10,
                    label="Max Tokens",
                )

            with gr.Row():
                submit_btn = gr.Button(
                    "Summarize",
                    variant="primary",
                    size="lg",
                )
                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary",
                    size="lg",
                )

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="الملخص بالدارجة / Darija Summary",
                placeholder="التلخيص هنا ",
                lines=8,
                max_lines=15,
                elem_classes="rtl-output",
                show_copy_button=True,
            )
    gr.Examples(
        examples=EXAMPLES,
        inputs=[prompt_input, temperature_slider, max_tokens_slider],
        label="أمثلة / Examples — click to load",
    )

    gr.Markdown("""
    ---
    **How it works:**  
    Arabic text → LLaMA-3.2-1B (finetuned on Darija) → Moroccan Arabic summary  

    **Stack:** QLoRA finetuning · vLLM serving · FastAPI · Gradio  
    **Hardware:** Trained on RTX 2060 6GB  

    *Built as part of an end-to-end MLOps project — [GitHub](https://github.com/yourname/darija-llmops)*
    """)

    submit_btn.click(
        fn=summarize_stream,
        inputs=[prompt_input, temperature_slider, max_tokens_slider],
        outputs=output,
        show_progress=False,   
    )
    prompt_input.submit(
        fn=summarize_stream,
        inputs=[prompt_input, temperature_slider, max_tokens_slider],
        outputs=output,
        show_progress=False,
    )
    clear_btn.click(
        fn=lambda: ("", ""),
        outputs=[prompt_input, output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        share=False,
    )