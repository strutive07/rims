def get_user_assistant_messages(
    system_message: str, user_message: str, assistant_message: str
):
    """
    This function is used to convert the prompt into the message format used by OpenAI Chat API.
    """
    messages = []
    messages.append({"role": "system", "content": system_message})
    split_user_messages = user_message.split("\n" * 4)
    split_assistant_messages = assistant_message.split("\n" * 4)  # delim==4*\n...
    for i in range(
        len(split_user_messages)
    ):  # user messages and assistant messages are paired... actually. This should have been `zip()`.
        question = split_user_messages[i]
        answer = split_assistant_messages[i]
        messages += [
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"{answer}"},
        ]
    return messages
