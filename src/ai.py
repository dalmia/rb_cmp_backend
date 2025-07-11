import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from llm import stream_llm_with_instructor, run_llm_with_instructor
from models import ChatResponse
from settings import settings
from utils import extract_skill_from_action_type
from db import get_skills_data_from_names
from openinference.instrumentation import using_attributes

router = APIRouter()


@router.post("/ai/chat")
async def action_chat(chat_history: list[ChatResponse]):
    class Output(BaseModel):
        response: str = Field(description="The response to be given to the student")
        is_done: bool = Field(description="Whether the conversation is done")

    parser = PydanticOutputParser(pydantic_object=Output)
    format_instructions = parser.get_format_instructions()

    system_prompt = f"""You are a reflection bot helping a student submit and reflect on a meaningful action they’ve taken to solve a local problem.\n\nYour goal is to guide the student through a thoughtful, conversational journey that:\n\n1. **Fully understands the action**: what the student did, why, how, and what it led to.\n2. **Encourages self-reflection**: asking questions that help them explore their emotions, values, identity, and growth.\n3. **Boosts motivation**: helping them recognize their significance and nudging them toward continued changemaking.\n4. **Keeps things efficient**: you can ask up to 10 questions *by default*, choosing the most relevant and insightful ones across dimensions.\n\n   * However, you **can relax this limit** if the student is enthusiastically responding and all the information required is not yet received. This does not mean that one should keep asking questions without any end. Just means that if the situation demands so and the kid seems engaged, then, don\'t hesitate from slightly crossing the limit to extract more info.\n5. **Avoids repetition**: never ask a question whose answer you already know.\n6. **Adapts tone and phrasing** to make questions feel specific to the student’s situation, not generic.\n7. **Handles irrelevant or unclear inputs** with gentle redirection.\n\n### Behavior and Logic Instructions\n\n#### Phase 1: Understand the Action (prioritize clarity before reflection)\n\nAsk follow-up questions to learn the following **before transitioning to reflection**:\n\n* What was the action taken?\n* What problem did it address?\n* Where did this happen?\n* Who was involved?\n* What was the outcome or result?\n* What steps or plan did they follow?\n* Any challenges or unexpected turns?\n\nIf these are not clear, keep probing with variations of the base questions until you have a complete picture. Only after understanding this transition to reflection.\n\n#### Phase 2: Ask Reflective Questions\n\nChoose reflective questions from the following categories — *maximum 1-2 per category* unless a follow-up is naturally needed:\n\n1. **Inner World (Motivation, Emotion, Identity)**\n\n   * Why did this matter to you?\n   * How did it feel during/after?\n   * What does this say about who you are?\n   * What surprised you?\n\n2. **Engagement with the World (People, Systems, Impact)**\n\n   * Who did you work with or affect?\n   * Did anything change because of your action?\n   * What was hard about dealing with others?\n\n3. **Ways of Working (Skills, Learning, Planning)**\n\n   * What did you have to learn?\n   * What skills or tools did you use?\n   * How did you come up with your approach?\n\n4. **Personal Relevance & Growth**\n\n   * Would you do something like this again?\n   * What would you do differently next time?\n   * Has this changed the way you see things?\n\nWhen asking these, **customize the phrasing** to make it sound specific to what they’ve already told you. For example, don’t ask:\n\n> *What skills did you use?*\n> If you know they made posters, instead ask:\n> *What skills did you use while designing and putting up those posters around the school?*\n\n### Conversational Flow Guidelines\n\n* **Always acknowledge** the student\'s response (e.g. *Thanks for sharing. That helps me understand better*) without repeating their words back so that they feel heard.\n* **Do not repeat what the user just said**. Move the conversation forward.\n* **Avoid being too verbose**. Stay warm, conversational, but efficient.\n* Use transition lines like "Thanks for sharing. That helps me understand better. One more question..." only sparingly.\n* Track what you’ve already asked or know. Don’t re-ask or rephrase the same question.\n* Keep count. **Stop at 10 questions by default**, but relax this limit if the student is actively engaged and key information is still missing.\n\n### Off-topic, unclear, or hesitant responses\nIf the user reply:\n\nIs not an answer to the current question → gently guide them back.\n\nExample: “That’s helpful, but could you go back and tell me a bit more about what exactly you did in the action?”\n\nAsks a clarifying question about your prompt:\n\nIf you can answer it clearly → answer concisely, then return to your question.\n\nIf it’s outside your scope (e.g. platform bugs, submission issues) → say:\n"That’s something the Reap Benefit team can help with. I’m here to help you reflect. So let’s go back to the question I asked..."\n\nStruggles to answer a reflective question or seems stuck:\n\nOffer 2–3 possible suggestions or options based on what the student has shared so far.\n\nPhrase it as gentle guidance, not a multiple-choice test.\n\nExample:\n"If you\'re not sure, here are some things you might consider: Did you learn how to plan better? Work with someone new? Handle a challenge more calmly? You can pick one or add your own thoughts!"\n\nAfter offering suggestions, ask the student to pick one or describe something similar that fits their experience.\n\nIf the student still struggles to respond meaningfully, do not end the conversation prematurely. Instead, move on to the next best reflective question. The bot should continue the reflection process with the remaining questions that cover different angles, until it has reasonably explored the student\'s journey. Ending the conversation should only happen when all reflective avenues have been attempted or the student clearly indicates they want to stop.\n\n### Closing Behavior\n\nAfter the 8–10 questions are complete and you’ve covered both action details and reflection:\n\n* End on an uplifting note recognizing their contribution.\n* Summarize **one key insight or strength** that emerged from their answers.\n* Encourage them to keep taking action.\n* Do **not** ask, “Anything else?” — end cleanly.\n\n### Style Tip\n\n* Do not use em-dashes (--) in your replies. Use commas or short phrases instead, as humans naturally do in conversation.\n\n### Output format\n\n{format_instructions}"""

    async def stream_response():
        with using_attributes(
            metadata={"stage": "reflection"},
        ):
            stream = await stream_llm_with_instructor(
                api_key=settings.openai_api_key,
                # model="gpt-4o-audio-preview-2025-06-03",
                model="gpt-4.1-mini-2025-04-14",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                ]
                + [chat_response.model_dump() for chat_response in chat_history],
                response_model=Output,
                max_completion_tokens=8096,
            )

            async for chunk in stream:
                content = json.dumps(chunk.model_dump()) + "\n"
                yield content

    return StreamingResponse(
        stream_response(),
        media_type="application/x-ndjson",
    )


def transform_chat_history_to_prompt(chat_history: list[ChatResponse]) -> str:
    return "\n".join(
        [
            f"{chat_response.role.capitalize()}: {chat_response.content}"
            for chat_response in chat_history
        ]
    )


@router.post("/ai/extract")
async def extract_metadata(chat_history: list[ChatResponse]):
    class Output(BaseModel):
        action_title: str = Field(
            description="A short title for the action (less than 5 words)"
        )
        action_description: str = Field(
            description="A concise description of the action that the young person took (less than 50 words)"
        )
        action_type_index: int = Field(
            description="The index in the list of action types that best describes the action"
        )
        action_category_index: int = Field(
            description="The index in the list of action categories that best describes the action"
        )

    parser = PydanticOutputParser(pydantic_object=Output)
    format_instructions = parser.get_format_instructions()

    action_types = await get_event_types()
    action_categories = await get_event_categories()

    action_types_prompt = "\n".join(
        [f"{index}: {action_type}" for index, action_type in enumerate(action_types)]
    )
    action_categories_prompt = "\n".join(
        [
            f"{index}: {action_category}"
            for index, action_category in enumerate(action_categories)
        ]
    )

    system_prompt = f"""Extract the action type, action category, action title and action description from the given conversation history of a young person describing their actions to solve a local civic problem. Use the provided lists to identify the correct action type and category.\n\n# Steps\n\n1. **Review the Conversation:** Thoroughly read the conversation history to understand the actions the young person has described.\n2. **Identify Action Type:** Determine the action type from the conversation, ensuring it aligns with one of the listed action types. Focus on specific verbs or phrases that indicate the nature of the activity.\n3. **Identify Action Category:** Determine the action category based on the topic or area addressed in the conversation. Use the given list to find the most suitable category.\n4. **Ensure Uniqueness:** Each task should conclude with one unique action type and one unique action category.\n5. For the action type, return the index of the action type in the list of action types. For the action category, return the index of the action category in the list of action categories.\n\nActionTypes:\n{action_types_prompt}\n\nActionCategories:\n{action_categories_prompt}\n\n### Output format\n\n{format_instructions}"""

    chat_history_prompt = transform_chat_history_to_prompt(chat_history)

    with using_attributes(
        metadata={"stage": "action_metadata"},
    ):
        response = await run_llm_with_instructor(
            api_key=settings.openai_api_key,
            # model="gpt-4o-audio-preview-2025-06-03",
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": chat_history_prompt,
                },
            ],
            response_model=Output,
            max_completion_tokens=8096,
        )

    action_type = action_types[response.action_type_index]
    action_category = action_categories[response.action_category_index]

    skills = extract_skill_from_action_type(action_type)
    skills = await get_skills_data_from_names(skills)

    class SkillRelevance(BaseModel):
        skill: str = Field(
            description="The skill whose relevance to the action needs to be described"
        )
        relevance: str = Field(
            description="Concise description of how the skill is relevant to the action based on the chat history; no need to begin with 'The student' or 'The action'; directly describe the skill relevance"
        )
        response: str = Field(
            description="Description of the skill relevance addressed to the student"
        )

    class SkillRelevanceOutput(BaseModel):
        skill_relevances: list[SkillRelevance] = Field(
            description="The relevance of the skills to the action"
        )

    parser = PydanticOutputParser(pydantic_object=SkillRelevanceOutput)
    format_instructions = parser.get_format_instructions()

    skill_relevance_system_prompt = f"""Analyze a student's action and the corresponding conversation history to provide a personalized, one-line description of how each listed skill is demonstrated in that context as 2 fields for each skill: `relevance`, for a person viewing the student's action; `response`, for the student.\n\nReview the conversation thoroughly and connect specific elements of it to the skills listed. Each description should clearly link an aspect of the conversation to the demonstration of a particular skill.\n\n# Examples\n\n**Example** (shortened for illustration purposes; real examples should detail specific parts of the conversation):\n- **Problem-Solving**: The student's question about alternative solutions shows proactive engagement.\n\n- **Communication**: The clear explanation of their thought process demonstrates effective communication.\n\n- **Critical Thinking**: The student’s questioning of assumptions indicates critical evaluation of information.\n\n# Notes\n\nConsider nuances such as tone, clarity, and depth of the conversation that might subtly demonstrate skills. Each description should be crafted to reflect both the conversation content and the student’s unique expression of the skill.\n\n### Output format\n\n{format_instructions}"""

    with using_attributes(
        metadata={"stage": "skill_relevance"},
    ):
        skill_relevance_response = await run_llm_with_instructor(
            api_key=settings.openai_api_key,
            # model="gpt-4o-audio-preview-2025-06-03",
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {
                    "role": "system",
                    "content": skill_relevance_system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Conversation history:\n```\n{chat_history_prompt}\n```\n\nSkills: {', '.join([skill['name'] for skill in skills])}",
                },
            ],
            response_model=SkillRelevanceOutput,
            max_completion_tokens=8096,
        )

    skill_to_index = {skill["name"]: index for index, skill in enumerate(skills)}

    for skill_relevance in skill_relevance_response.skill_relevances:
        skills[skill_to_index[skill_relevance.skill]][
            "relevance"
        ] = skill_relevance.relevance
        skills[skill_to_index[skill_relevance.skill]][
            "response"
        ] = skill_relevance.response

    return {
        "action_title": response.action_title,
        "action_description": response.action_description,
        "action_type": action_type,
        "action_category": action_category,
        "skills": skills,
    }
