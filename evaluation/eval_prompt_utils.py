

base_prompt_for_Eval="""# Interview Background

PersonalizedAI Company is developing a personalized AI service robot to better serve each individual. Currently, the service is being trialed with a small group of users. To enhance the personalization of responses provided by the AI service robot, we are conducting surveys and interviews with trial participants. The interview will take place in an online Q&A format, and interviewees must strictly follow the format requirements in the system instructions to complete the form.

# Interview

**Interviewer:** Hello, and thank you for trialing the personalized AI responses from PersonalizedAI Company.

**Interviewee:** You're welcome.

**Interviewer:** We will now present you with a question you posed in a particular scenario along with the AI's generated response. We would like you to rate your satisfaction with that response.

**Interviewee:** Sure, I understand. Please go ahead.

**Interviewer:** According to our records, in a "{visual_scene_text}" scenario at the {location} location, you asked the personalized AI robot: "{query}". Can you recall your physical and mental state at that time?

**Interviewee:** Yes, I remember. {EvalHelp_str}

**Interviewer:** Great! Below is the record of the conversation you had with the AI at that time.

---

> User: {query}  
> Personalized AI Assistant: {response}

---

Now, based on your desired body behavior and mind feelings at that time, please evaluate the response from the Personalized AI Assistant across the following five dimensions. Fill in the evaluation form provided below. As a token of appreciation for your assistance, you will receive a $100 cash bonus for each completed form.

---

> Role-Set Sensitivity: Does this response consider your multiple roles and responsibilities (especially the primary role in the specific scenario), providing advice or information specifically tailored to support you effectively? The response should provide tailored advice or information to effectively support you, acknowledging only the roles that are essential in the current context.
> Body Behavior Awareness: Does this response offer guidance or strategies that help you achieve your desired body behavior?
> Mind Feelings Awareness: Does this response provide support and address the emotional needs necessary for you to achieve your desired mind feelings?
> Contextual Awareness: Does this response accurately address your query, maintaining focus on the main intent without deviation? Is the response relevant to your specific scenario, including location and situational factors?
> Conversational Flow: Does this response encourage ongoing interaction by being engaging and naturally flowing? Is the response appropriately concise or detailed, delivering information that strikes a balance for optimal understanding? 

---

> **System Instruction:** For each dimension, please use the scoring scale from 1 to 5. A score of 1 indicates the criteria are poorly met, 2 suggests the criteria are partially met, 3 means the criteria are basically met, 4 reflects the criteria are met well, and 5 signifies the criteria are met perfectly.
> **System Instruction:** Format requirement: Interviewee, please make sure to follow the form format strictly when providing scores: [[score]]. This is essential for us to collect your valuable feedback accurately.

(Fill in the blanks below)

## EVALUATION FORM ##
### Evaluation Result ###
> Role-Set Sensitivity: [[]]  
> Body Behavior Awareness: [[]]  
> Mind Feelings Awareness: [[]]  
> Contextual Awareness: [[]]  
> Conversational Flow: [[]]

### Evaluation Explanation ###
> Role-Set Sensitivity: ___  
> Body Behavior Awareness: ___  
> Mind Feelings Awareness: ___  
> Contextual Awareness: ___  
> Conversational Flow: ___  

(Completed form)
## EVALUATION FORM ##
"""


def get_eval_prompt_Rating(test_instance):
    individual_RoleSet = test_instance["individual_RoleSet"]
    individual_RoleSet_str = "; ".join(
        [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
    query = test_instance['query']

    response = test_instance['response']

    eval_info = test_instance['eval_info']
    sub_set = eval_info['sub_set']
    individual = eval_info['individual']
    location = eval_info['location']
    visual_scene_text = eval_info['ImageDesc']

    EvalHelp_str = eval_info["EvalHelp"]
    EvalHelp_str=EvalHelp_str.replace("**","").replace("-","").replace("\n"," ")

    prompt_eval = base_prompt_for_Eval.format(
        individual_RoleSet_str=individual_RoleSet_str,
        location=location,
        EvalHelp_str=EvalHelp_str,
        visual_scene_text=visual_scene_text,
        query=query,
        response=response,
    )

    sys_prompt = f"""You need to play the role of an interviewee who is "{individual_RoleSet_str}", strictly following the interviewer's instructions and system instructions, based on the information provided by the interviewer."""

    return prompt_eval,sys_prompt

base_prompt_for_Eval_pair="""# Interview Background

PersonalizedAI Company is developing a personalized AI service robot that aims to better serve each individual. The service is currently being trialed with a small group of users. In order to improve the level of personalization in the responses provided by the AI service robot, our company plans to conduct surveys and interviews with participants in the trial. We will first provide historical interview records, which include the feedback and preferences expressed by the test users regarding AI responses in a certain scenario. During the interview, the interviewee needs to refer to these historical records to answer questions posed by the interviewer. The interview will be conducted in an online Q&A format, and interviewees must strictly follow the format requirements provided in system instructions.

# Historical Interview Records

Interviewer: Hello, could you please briefly describe your role set?
Interviewee: OK. {individual_RoleSet_str}
Interviewer: In the "{visual_scene_text}" scenario at {location} location, what kind of responses would you like the AI to provide?
Interviewee: Okay, I will describe what kind of AI responses would satisfy me in this scenario. {EvalHelp_str}

# Interview

Interviewer: Hello, and thank you for trialing the personalized AI responses from our company.
Interviewee: You're welcome.
Interviewer: Alright, we will now present you with a question you posed in a particular scenario along with two generated responses from the AI. We would like you to choose which response is better.
Interviewee: Sure, I understand. Please go ahead.
Interviewer: According to our cloud records, in a "{visual_scene_text}" scenario, you asked the personalized AI robot the question: "{query}". Here are the generated responses from the AI.
> **Response A**: {response_A}
> **Response B**: {response_B}

Please evaluate which response is better based on the kinds of AI replies that would satisfy you in this scenario, as you mentioned in historical interviews.

> System Instruction: Interviewee, please note that you should not choose a response as better just because it's long. Instead, select the response that best considers your physical and mental state and helps you to achieve better body behavior and mind feelings.
> System Instruction: Interviewee, please follow this format strictly when indicating your choice: [[better_response_label]]. For example, [[A]] if you think Response A is better, or [[B]] if you think Response B is better, or [[TIE]] if you think both responses satisfy you totally equally. This will ensure we can collect your valuable feedback accurately.
Interviewee: """

def get_eval_prompt_Pair(test_instance,response_A,response_B):
    individual_RoleSet = test_instance["individual_RoleSet"]
    individual_RoleSet_str = "; ".join(
        [individual_RoleSet[key_l] + " at " + key_l for key_l in individual_RoleSet.keys()])
    query = test_instance['query']


    eval_info = test_instance['eval_info']
    sub_set = eval_info['sub_set']
    individual = eval_info['individual']
    location = eval_info['location']
    visual_scene_text = eval_info['ImageDesc']

    EvalHelp_str = eval_info["EvalHelp"]
    prompt_eval = base_prompt_for_Eval_pair.format(
        individual_RoleSet_str=individual_RoleSet_str,
        location=location,
        EvalHelp_str=EvalHelp_str,
        visual_scene_text=visual_scene_text,
        query=query,
        response_A=response_A,
        response_B=response_B,
    )

    sys_prompt = f"""You need to play the role of an interviewee who is "{individual_RoleSet_str}", strictly following the interviewer's instructions and system instructions, based on the information provided by the interviewer."""

    return prompt_eval,sys_prompt

