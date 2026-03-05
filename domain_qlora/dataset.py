from typing import List, Dict, Tuple

class InstructionDataset:
    """Domain-specific instruction dataset with Llama-3 ChatML template formatting."""
    def __init__(self):
        self.data: List[Dict[str, str]] = [
            {
                "instruction": "What are the contraindications for administering beta-blockers in acute myocardial infarction?",
                "input": "Patient has severe bradycardia (HR < 45 bpm) and second-degree AV block.",
                "output": "Beta-blockers are contraindicated in severe bradycardia, second- or third-degree heart block, cardiogenic shock, and severe decompensated heart failure due to risk of complete heart block and worsening hemodynamics."
            },
            {
                "instruction": "Explain the pharmacokinetic impact of renal impairment on metformin clearance.",
                "input": "eGFR < 30 mL/min/1.73m2",
                "output": "Metformin is excreted unchanged by the kidneys. Severe renal impairment markedly reduces renal clearance, leading to drug accumulation and substantially increased risk of lactic acidosis. Discontinuation is required when eGFR drops below 30."
            },
            {
                "instruction": "How should an acute asthma exacerbation be managed pharmacologically?",
                "input": "Patient presenting with wheezing, peak expiratory flow < 50% predicted.",
                "output": "First-line therapy: High-dose inhaled short-acting beta-2 agonist (albuterol) combined with ipratropium bromide via nebulizer, along with early systemic corticosteroids (oral prednisone or IV methylprednisolone) and supplemental oxygen."
            }
        ]

    def format_chatml(self, item: Dict[str, str]) -> str:
        prompt = f"<|im_start|>system\nYou are an expert clinical pharmacology AI assistant.<|im_end|>\n"
        prompt += f"<|im_start|>user\n{item['instruction']}\nContext: {item['input']}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{item['output']}<|im_end|>"
        return prompt

    def get_tokenized_batches(self) -> List[str]:
        return [self.format_chatml(item) for item in self.data]
