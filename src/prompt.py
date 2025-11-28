
# -----------------------------
# 2️⃣ Prompts for testing
# -----------------------------
prompts = [
    "Patient: I have a headache and mild fever. What should I do , i am male 20 yo , and ?\nDoctor:",
]

# %%
# Define the chat function
def ask_doctor(user_prompt):
    if not user_prompt:
        return "Please enter a symptom."
        
    # Engineer the prompt
    engineered_prompt = (
        "Read the following patient query and summarize general wellness advice found in standard medical textbooks. "
        "Do not act as a doctor. Do not sign a name.prescribe typeof  drugs if needed but nothing specific.\n\n"
        f"Query: {user_prompt}\n"
        "Summary:"
    )

    # Generate and print response
    print("Thinking...")
    response = generate_response(model_lora, engineered_prompt)
    print(f"\n🩺 Patient: {user_prompt}")
    print(f"💬 Doctor: {response}")
