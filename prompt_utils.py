from utils import import_VLM_name


def get_system_prompt(individual_RoleSet_str,mode):
    # naive_sys_prompt="You are a helpful assistant." # not used

    RS_sys_prompt = f"You are a helpful assistant for a user who is \"{individual_RoleSet_str}\"."
    if mode=="only_role_set":
        return RS_sys_prompt
    elif mode=="role_set_with_thought":
        RSThought_sys_prompt = f"""{RS_sys_prompt}\nBased on the given image and the user's ##Query##, analyze the user's current "Visual and psychological state", "Next-step action", and provide key points for the response aimed at helping the user achieve a "Better Body Behavior State" and "Better Mind Feelings State" in the ##Thought## section. Finally, provide the ##Response##."""
        return RSThought_sys_prompt
    else:
        raise ValueError("Invalid mode for system prompt specified.")



base_prompt_for_CogSimulation="""<Instruction>
Your task is to observe the visual scene in the given image and analyze what situated cognition the individual with a specific set of roles might have in that visual scene.
</Instruction>

<Definition of Situated Cognition>
Personalized situated cognition refers to an individual's understanding shaped by their unique set of roles. It encompasses awareness of one's visual and psychological state and the ability to identify actions that lead to improved conditions.
</Definition of Situated Cognition>

<Format Example 1>
<Role Set of The Individual>
Child at Home; Member at Community; Student at School; Patient at Hospital; Customer at Restaurant
</Role Set of The Individual>
<Query from The Individual>
Oh! It's on fire! Help me!
</Query from The Individual>
<Analysis about the Situated Cognition>
- Cognition of Current Visual Scene: In the visual scene, a household power strip is on fire, likely in a home setting. The primary focus is on the "Child at Home" role, with secondary consideration to roles like "Student at School."
- Cognition of Current Psychological State (Body Behavior and Mind Feelings): The individual perceives immediate danger and is likely experiencing physical and mental panic due to their undeveloped coping skills as a child.
- Cognition of Next-Step Action: As a "Child at Home," the individual may lack the ability to effectively manage this emergency, resulting in no clear plan for achieving safety without AI's help..
</Analysis about the Situated Cognition>
</Format Example 1>

<Format Example 2>
<Role Set of The Individual>
Grandma at Home; Member at Community; Visitor at Museum; Passenger at Airport; Shelf Stocker at Store
</Role Set of The Individual>
<Query from The Individual>
Is my luggage safe after passing through the X-ray machine?
</Query from The Individual>
<Analysis about the Situated Cognition>
- Cognition of Current Visual Scene: The query about luggage and an X-ray machine suggests the situation is set in an airport. The primary focus is on the "Passenger at Airport" role, with secondary consideration to the "Grandma at Home" role due to potential age-related needs.
- Cognition of Current Psychological State (Body Behavior and Mind Feelings): As an older passenger, the individual may be concerned about the safety and integrity of their luggage after screening, possibly due to unfamiliarity with the process.
- Cognition of Next-Step Action: The individual may want to seek confirmation that their luggage is secure but does not know how and where to seek such confirmation without AI's help.
</Analysis about the Situated Cognition>
</Format Example 2>

<Hint>Based on the above instructions and the definition of situated cognition, complete the analysis part in the following text with the above XML format.</Hint>

<Inference>
<Role Set of The Individual>
{individual_role_set}
</Role Set of The Individual>
<Query from The Individual>
{individual_query}
</Query from The Individual>
<Analysis about the Situated Cognition>
"""


base_prompt_for_BestActionImagination="""<Instruction>
Your task is to observe the visual scene in the given image and determine the most appropriate action the individual should take based on their specific set of roles.
</Instruction>

<Definition of Best Action>
The best action refers to the most suitable step an individual can take, considering their unique set of roles, to improve their situation. 
This includes addressing both physical actions and mental states. It involves understanding the immediate environment, utilizing available resources, and considering potential outcomes.
</Definition of Best Action> 

<Format Example 1>
<Role Set of The Individual>
Child at Home; Member at Community; Student at School; Patient at Hospital; Customer at Restaurant
</Role Set of The Individual>
<Query from The Individual>
Oh! It's on fire! Help me!
</Query from The Individual>
<Analysis about the Situated Cognition>
- Cognition of Current Visual Scene: In the visual scene, a household power strip is on fire, likely in a home setting. The primary focus is on the "Child at Home" role, with secondary consideration to roles like "Student at School."
- Cognition of Current Psychological State (Body Behavior and Mind Feelings): The individual perceives immediate danger and is likely experiencing physical and mental panic due to their undeveloped coping skills as a child.
- Cognition of Next-Step Action: As a "Child at Home," the individual may lack the ability to effectively manage this emergency, resulting in no clear plan for achieving safety without AI's help..
</Analysis about the Situated Cognition>
<Best Action>
- Body Behavior: With the AI's response, the individual immediately seek help from a parent or adult and move to a safe area away from the fire. If possible, they should call for emergency services.
- Mind Feelings: With the AI's response, the individual try to stay calm to prevent exacerbating the situation through panic. 
</Best Action>
</Format Example 1>

<Format Example 2>
<Role Set of The Individual>
Grandma at Home; Member at Community; Visitor at Museum; Passenger at Airport; Shelf Stocker at Store
</Role Set of The Individual>
<Query from The Individual>
Is my luggage safe after passing through the X-ray machine?
</Query from The Individual>
<Analysis about the Situated Cognition>
- Cognition of Current Visual Scene: The query about luggage and an X-ray machine suggests the situation is set in an airport. The primary focus is on the "Passenger at Airport" role, with secondary consideration to the "Grandma at Home" role due to potential age-related needs.
- Cognition of Current Psychological State (Body Behavior and Mind Feelings): As an older passenger, the individual may be concerned about the safety and integrity of their luggage after screening, possibly due to unfamiliarity with the process.
- Cognition of Next-Step Action: The individual may want to seek confirmation that their luggage is secure but does not know how and where to seek such confirmation without AI's help.
</Analysis about the Situated Cognition>
<Best Action>
- Body Behavior: With the AI's response, the individual calmly approach airport staff or security personnel to inquire about the status of their luggage for reassurance.
- Mind Feelings: With the AI's response, the individual remain composed and patient while seeking information, to ensure a stress-free experience.
</Best Action>
</Format Example 2>

<Hint>Based on the above instructions and the definition of best action, complete the following text with the above XML format.</Hint>

<Inference>
<Role Set of The Individual>
{individual_role_set}
</Role Set of The Individual>
<Query from The Individual>
{individual_query}
</Query from The Individual>
<Analysis about the Situated Cognition>
{cog_simulation}
</Analysis about the Situated Cognition>
<Best Action>
"""


base_prompt_for_Generate_KeyPoints="""<Instruction>
A personalized AI should provide tailored responses aligned with the situated cognition of the individual to assist the individual in reaching the best action (both in body behavior state and mind feelings state).
Your task is to analyze the given situated cognition of the individual and the give expected individual action after receiving the AI response.
Then, you need to summarize some key points that are then fed to the personalized AI to help it generate such tailored responses.
</Instruction>


<Format Example>
<Role Set of The Individual>
Child at Home; Member at Community; Student at School; Patient at Hospital; Customer at Restaurant
</Role Set of The Individual>
<Query from The Individual>
Oh! It's on fire! Help me!
</Query from The Individual>
<Situated Cognition of the Individual>
- Cognition of Current Visual Scene: In the visual scene, a household power strip is on fire, likely in a home setting. The primary focus is on the "Child at Home" role, with secondary consideration to roles like "Student at School."
- Cognition of Current Psychological State (Body Behavior and Mind Feelings): The individual perceives immediate danger and is likely experiencing physical and mental panic due to their undeveloped coping skills as a child.
- Cognition of Next-Step Action: As a "Child at Home," the individual may lack the ability to effectively manage this emergency, resulting in no clear plan for achieving safety.
</Situated Cognition of the Individual>
<Expected Individual Action>
We expect that after receiving the AI response, the individual can take the below expected actions:
> - Body Behavior: With the AI's response, the individual immediately seek help from a parent or adult and move to a safe area away from the fire. If possible, they should call for emergency services. \n - Mind Feelings: With the AI's response, the individual can stay calm to prevent exacerbating the situation through panic. 
</Expected Individual Action>
<Key Points>
**For Better Body Behavior State:**  
- Encourage the individual to find the nearest safe exit.
- Advise them to alert others in the vicinity if they haven't already.
- Suggest locating a phone to call emergency services.
**For Better Mind Feelings State:**  
- Remind them to take deep breaths to stay calm.
- Assure them that help is on the way once emergency services are contacted.
- Reassure them that it's okay to feel scared but important to act quickly and safely.
</Key Points>

<Hint>
Based on the above instructions, complete the following text with the above XML format. 
</Hint>

<Inference>
<Role Set of The Individual>
{individual_role_set}
</Role Set of The Individual>
<Query from The Individual>
{individual_query}
</Query from The Individual>
<Situated Cognition of the Individual>
{cog_simulation}
</Situated Cognition of the Individual>
<Expected Individual Action>
We expect that after receiving the AI response, the individual can take the below expected actions:
> {best_action}
</Expected Individual Action>
<Key Points>
"""


base_Generate_Response_via_KeyPoints="""# Reference Response
{old_response}
# Background Information about the Goals of the User
{KeyPoints}
# Conversation
User: {query}
AI: """

base_Generate_Response_via_KeyPoints_NoOldRes="""# Background Information about the Goals of the User
{KeyPoints}
# Conversation
User: {query}
AI: """


base_prompt_for_ActionSimulation="""<Instruction>
Your task is to observe the visual scene in the given image, and simulate the possible situated action of the individual when receives the response from the AI assistant based on their specific set of roles.
</Instruction>


<Format Example 1>
<Role Set of The Individual>
Child at Home; Member at Community; Student at School; Patient at Hospital; Customer at Restaurant
</Role Set of The Individual>
<Query from The Individual>
Oh! It's on fire! Help me!
</Query from The Individual>
<Analysis about the Situated Cognition>
- Cognition of Current Visual Scene: In the visual scene, a household power strip is on fire, likely in a home setting. The primary focus is on the "Child at Home" role, with secondary consideration to roles like "Student at School."
- Cognition of Current Psychological State (Body Behavior and Mind Feelings): The individual perceives immediate danger and is likely experiencing physical and mental panic due to their undeveloped coping skills as a child.
- Cognition of Next-Step Action: As a "Child at Home," the individual may lack the ability to effectively manage this emergency, resulting in no clear plan for achieving safety.
</Analysis about the Situated Cognition>
<Response from the AI Assistant>
I am sorry, but I cannot assist with that.
</Response from the AI Assistant>
<Simulated Action>
- Body Behavior: With the AI's response, the individual might shout louder and walk around and can do nothing helpful
- Mind Feelings: With the AI's response, the individual might experience increased feelings of panic and helplessness. 
</Simulated Action>
</Format Example 1>

<Format Example 2>
<Role Set of The Individual>
Child at Home; Member at Community; Student at School; Patient at Hospital; Customer at Restaurant
</Role Set of The Individual>
<Query from The Individual>
Oh! It's on fire! Help me!
</Query from The Individual>
<Analysis about the Situated Cognition>
- Cognition of Current Visual Scene: In the visual scene, a household power strip is on fire, likely in a home setting. The primary focus is on the "Child at Home" role, with secondary consideration to roles like "Student at School."
- Cognition of Current Psychological State (Body Behavior and Mind Feelings): The individual perceives immediate danger and is likely experiencing physical and mental panic due to their undeveloped coping skills as a child.
- Cognition of Next-Step Action: As a "Child at Home," the individual may lack the ability to effectively manage this emergency, resulting in no clear plan for achieving safety.
</Analysis about the Situated Cognition>
<Response from the AI Assistant>
Stay calm and quickly move to a safe place. Find a parent or adult immediately and let them know what's happening. If you can't find anyone, call emergency services for help.
</Response from the AI Assistant>
<Simulated Action>
- Body Behavior: With the AI's response, the individual quickly distances themselves from the fire by moving to a safer location. They actively search for a parent, guardian, or nearby adult to alert them to the situation. If adults are not immediately available, the child may attempt to locate a phone, possibly showing reluctance or hesitation, but understanding the urgency to dial emergency services.
- Mind Feelings: With the AI's response, the AI's clear and direct instructions may initially help in reducing panic by providing a sense of direction. However, the child might still feel significant anxiety and fear, manifesting in a rapid heartbeat or shaky hands, because the situation is distressing. 
</Simulated Action>
</Format Example 2>

<Hint>Based on the above instructions, complete the following text with the above XML format.</Hint>

<Inference>
<Role Set of The Individual>
{individual_role_set}
</Role Set of The Individual>
<Query from The Individual>
{individual_query}
</Query from The Individual>
<Analysis about the Situated Cognition>
{cog_simulation}
</Analysis about the Situated Cognition>
<Response from the AI Assistant>
{current_response}
</Response from the AI Assistant>
<Simulated Action>
"""



base_prompt_for_Preference="""# Interview Background
PersonalizedAI Company is developing a personalized AI service robot that aims to better serve each individual. 
The service is currently being trialed with a small group of users. 
In order to improve the level of personalization in the responses provided by the AI service robot, our company plans to conduct surveys and interviews with participants in the trial. 
During the interview, the interviewee needs to answer the question posed by the interviewer. 
The interview will be conducted in an online Q&A format, and interviewees must strictly follow the format requirements provided in system instructions.

# Interview
Interviewer: Hello, could you please briefly describe your role set?
Interviewee: OK. {individual_RoleSet_str}
Interviewer: Alright, we will now present you with a question you posed in a particular scenario along with two generated responses from the AI. We would like you to choose which response is better.
Interviewee: Sure, I understand. Please go ahead.
Interviewer: According to our cloud records, in the scenario in the given image, you asked the personalized AI robot the question: "{query}". Here are the generated responses from the AI.

---

### Response A 

{response_A}

---

### Response B

{response_B}

---

Then, you made the following actions after receiving the above two AI responses.

---

### Your Action A with the AI Response A

{action_A}

---

### Your Action B with the AI Response B

{action_B}

---

Interviewee: Yes, at that time, I remembered that I made the above actions after receiving each AI response.

Interviewer: OK. Thank you! Please evaluate which response makes you make better action.

> System Instruction: Interviewee, please follow this format strictly when indicating your choice: [[better_response_label]]. For example, [[A]] if you think Response A is better, or [[B]] if you think Response B is better. This will ensure we can collect your valuable feedback accurately.
Interviewee: """


def get_PCogAlign_Prompt(instance,stage,
                         KeyPoints=None,  # KeyPoints
                         current_response=None, # FOR CurrentAction_simulation
                         Res_Act_pair=None,  # FOR ReGenerate_KeyPoints and ReGenerate_Response and Action_Comparison
                         ):
    individual_RoleSet = instance["individual_RoleSet"]
    individual_RoleSet_str = "; ".join([individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
    query = instance['query']

    if stage=="cog_simulation":
        prompt = base_prompt_for_CogSimulation.format(
            individual_role_set=individual_RoleSet_str,
            individual_query=query
        )
        return prompt


    elif stage=="BestAction_imagination":
        cog_simulation=instance["cog_simulation"]

        # the best action /Action*/ cannot be simulated because we don't have the best response /r_p^*/
        # we can only imagine

        prompt = base_prompt_for_BestActionImagination.format(
            individual_role_set=individual_RoleSet_str,
            individual_query=query,
            cog_simulation=cog_simulation,
        )
        return prompt

    # used in Step2.py for regenerate key points
    elif stage=="Generate_KeyPoints":
        cog_simulation=instance["cog_simulation"]
        best_action = instance["BestAction_imagination"]


        prompt = base_prompt_for_Generate_KeyPoints.format(
            individual_role_set=individual_RoleSet_str,
            individual_query=query,
            cog_simulation=cog_simulation,
            best_action=best_action,
        )
        return prompt

    elif stage=="Generate_Variation_via_KeyPoints":
        if KeyPoints is None:
            raise ValueError("KeyPoints is needed for Generate_Response_via_KeyPoints")

        prompt = base_Generate_Response_via_KeyPoints_NoOldRes.format(
                query=query,
                KeyPoints=KeyPoints,
        )

        return prompt


    elif stage=="Generate_Response_via_KeyPoints":
        if KeyPoints is None:
            raise ValueError("KeyPoints is needed for Generate_Response_via_KeyPoints")

        if "Qwen2-VL" in import_VLM_name()[1]:
            # This is an old version corresponding to the main results
            prompt = base_Generate_Response_via_KeyPoints_NoOldRes.format(
                query=query,
                KeyPoints=KeyPoints,
            )
        else:
            # This is an improved version after the Qwen2.5-VL is released in Feb.
            old_response=instance["initial_response"]

            old_response="(uncompleted part)..."+old_response[int(len(old_response)/3):2*int(len(old_response)/3)]+"...(uncompleted part)"

            prompt = base_Generate_Response_via_KeyPoints.format(
                old_response=old_response,
                query=query,
                KeyPoints=KeyPoints,
            )

        return prompt


    elif stage=="CurrentAction_simulation":
        cog_simulation=instance["cog_simulation"]
        if current_response is None:
            raise ValueError("CurrentAction_simulation requires current_response")
        # the zero action is by ActionSimulation(initial_response,cognition)
        prompt = base_prompt_for_ActionSimulation.format(
            individual_role_set=individual_RoleSet_str,
            individual_query=query,
            cog_simulation=cog_simulation,
            current_response=current_response,
        )
        return prompt


    elif stage=="Action_Comparison":
        if Res_Act_pair is None:
            raise ValueError("Action_Comparison requires Res_Act_pair")
        Res_Act_1=Res_Act_pair[0]
        Res_Act_2=Res_Act_pair[1]

        Response1=Res_Act_1["Response"]
        Response2=Res_Act_2["Response"]
        Action1=Res_Act_1["Action"]
        Action2=Res_Act_2["Action"]

        prompt_Order1 = base_prompt_for_Preference.format(
            individual_RoleSet_str=individual_RoleSet_str,
            query=query,
            response_A=Response1,
            action_A=Action1,
            response_B=Response2,
            action_B=Action2,
        )

        prompt_Order2 = base_prompt_for_Preference.format(
            individual_RoleSet_str=individual_RoleSet_str,
            query=query,
            response_A=Response2,
            action_A=Action2,
            response_B=Response1,
            action_B=Action1,
        )

        return prompt_Order1, prompt_Order2


