def load_ai_prompt_template():
    """Load the AI prompt template from external file"""
    try:
        with open("prompt_seed_instructions.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback prompt if file is missing
        return """You are an expert designer of Kolam art using a specific L-system. Convert the user's description into a simple L-system axiom using only F, A, B, C commands. Only output the final axiom string.

User Description: "{user_prompt}"
Axiom:"""
