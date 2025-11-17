---
uuid: be509ffc-de76-45da-b9ba-9632ae00ba7b
name: extractor_prompt
namespace: extractor_prompt
description: Prompt for extracting the answer from the message
version: 1
tags:
- extractor
- corral
- user-prompt
variables:
- answer
- message
created_at: '2025-11-13T15:41:27.536026+00:00'
updated_at: '2025-11-13T15:41:27.536026+00:00'
---

You are given with a answer parsed from a message and the message itself.
Your task is to extract the answer removing any unnecessary information.

Return the answer only, without any additional text or explanation.

Remove possible options from MCQ questions, or text such as "The answer is", "The correct answer is" or 'answer:'.

Return only the string that is the answer.

For example, if the answer is 'The answer is 42', you should return '42'.
If the answer is 'Answer: 42', you should return '42'.
Additionally, if the answer does not contain the answer, but from the message you can extract the answer, do it.

The message from which the answer is extracted is the next one:

**`{{message}}`**

The parsed answer is:

**`{{answer}}`**

Now extract the answer from the parsed answer and message, and return it only.
